import numpy as np
import torch
from torch import nn
from typing import Optional, Union
import ncps
from . import CfCCell, LTCCell
from .lstm import LSTMCell


class LTC(nn.Module):
    def __init__(
            self,
            input_size: int,
            units,
            return_sequences: bool = True,
            batch_first: bool = True,
            mixed_memory: bool = False,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,
            implicit_param_constraints=True,
            use_wb_equations=True,
    ):
        """Applies a `Liquid time-constant (LTC)` RNN to an input sequence."""

        super(LTC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.use_wb_equations = use_wb_equations

        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)

        # Make sure you use the correct input dimensions when wiring.build()
        if not wiring.is_built():
            wiring.build(input_size)

        # Create an LTC unit, passing the use_wb_equations parameter
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
            use_wb_equations=use_wb_equations,
        )
        self._wiring = wiring
        self.use_mixed = mixed_memory

        # If using mixed memory, make sure the LSTM receives input of the correct dimensions
        if self.use_mixed:
            # LSTM should receive inputs of the same dimensions as the actual inputs, rather than always assuming a level-balanced output
            self.lstm = LSTMCell(input_size, self.state_size_without_wb())

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    def state_size_without_wb(self):
        """Returns the size of the neuron state excluding the water balance state"""
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    def forward(self, input, hx=None, timespans=None):
        """
        Forward Propagation Method

        parameter:
        input: input tensor, shape is (L, C) (no batch mode) or (B, L, C) (batch_first=True) or (L, B, C) (batch_first=False)
        hx: initial hidden state of RNN
        timespans: optional time step tensor

        return:
        (output, hx) tuple, where output is the RNN output and hx is the final hidden state
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        # Initialization state, different processing is performed depending on whether the water balance equation is used
        if hx is None:
            if self.use_wb_equations:
                # The water balance state requires 4 additional state variables
                total_state_size = self.state_size
                h_state = torch.zeros((batch_size, total_state_size), device=device)
            else:
                h_state = torch.zeros((batch_size, self.state_size_without_wb()), device=device)

            c_state = (
                torch.zeros((batch_size, self.state_size_without_wb()), device=device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a model with mixed_memory=True requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = (
                        "For batched 2-D input, hx and cx should "
                        f"also be 2-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
            else:
                # batchless mode
                if h_state.dim() != 1:
                    msg = (
                        "For unbatched 1-D input, hx and cx should "
                        f"also be 1-D but got ({h_state.dim()}-D) tensor"
                    )
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        wb_output = None

        for t in range(seq_len):
            if self.batch_first:
                inputs_t = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs_t = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            # Different treatments are performed depending on whether the water balance equation is used
            if self.use_wb_equations:
                # Separating neuronal state and water balance from hidden state
                neuron_state = h_state[:, :self.state_size_without_wb()]
                wb_state = h_state[:, self.state_size_without_wb():]

                # Only the first 4 features are used to calculate water balance
                wb_inputs = inputs_t[:, :4]

                # Calculate water balance output and obtain new water balance status
                wb_output, new_wb_state = self.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)

                # Append the water balance output to the input, creating an enhanced input
                augmented_inputs = torch.cat([inputs_t, wb_output], dim=1)

                # Update the water balance state part in h_state
                h_state_new = h_state.clone()
                h_state_new[:, self.state_size_without_wb():] = new_wb_state
                h_state = h_state_new
            else:
                # When water balancing is not used, the original input is used directly
                augmented_inputs = inputs_t

            # Hybrid memory mode processing
            if self.use_mixed:
                if self.use_wb_equations:
                    # Extracting neuron states
                    neuron_state = h_state[:, :self.state_size_without_wb()]

                    # Use LSTM processing to ensure the input dimension is correct
                    h_lstm, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                    # Only update the neuron part of h_state to keep the water balance state
                    h_state_new = h_state.clone()
                    h_state_new[:, :self.state_size_without_wb()] = h_lstm
                    h_state = h_state_new
                else:
                    # LSTM update without water balancing
                    h_state, c_state = self.lstm(inputs_t, (h_state, c_state))

            # Forward propagation LTC unit
            h_out, h_state = self.rnn_cell.forward(
                augmented_inputs if self.use_wb_equations else inputs_t,
                h_state,
                ts
            )

            if self.return_sequences:
                output_sequence.append(h_out)

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            # batchless mode
            readout = readout.squeeze(batch_dim)
            if self.use_mixed:
                hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]
            else:
                hx = h_state[0]

        return readout, hx
