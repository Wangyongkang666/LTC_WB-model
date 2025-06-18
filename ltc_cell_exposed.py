import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


class LTCCell(nn.Module):
    def __init__(
            self,
            wiring,
            in_features=None,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,
            implicit_param_constraints=False,
            use_wb_equations=True,  # Add parameter to control whether to use water balance equation
    ):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.LTC`.
        """
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'."
            )
        self.make_positive_fn = (
            nn.Softplus() if implicit_param_constraints else nn.Identity()
        )
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self._use_wb_equations = use_wb_equations  # Whether to use the water balance equation
        self._allocate_parameters()

        # Water balance equation parameters
        if self._use_wb_equations:
            self._init_wb_parameters()

    @property
    def state_size(self):
        # State size now includes neuron state and water balance state (if enabled)
        if self._use_wb_equations:
            return self._wiring.units + 1
        else:
            return self._wiring.units

    @property
    def regular_state_size(self):
        # Original size (excluding water balance)
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

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix))

    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak", init_value=self._get_init_value((self.regular_state_size,), "gleak")
        )
        self._params["vleak"] = self.add_weight(
            name="vleak", init_value=self._get_init_value((self.regular_state_size,), "vleak")
        )
        self._params["cm"] = self.add_weight(
            name="cm", init_value=self._get_init_value((self.regular_state_size,), "cm")
        )
        self._params["sigma"] = self.add_weight(
            name="sigma", init_value=self._get_init_value((self.regular_state_size, self.regular_state_size), "sigma"),
        )
        self._params["mu"] = self.add_weight(
            name="mu", init_value=self._get_init_value((self.regular_state_size, self.regular_state_size), "mu"),
        )
        self._params["w"] = self.add_weight(
            name="w", init_value=self._get_init_value((self.regular_state_size, self.regular_state_size), "w"),
        )
        self._params["erev"] = self.add_weight(
            name="erev", init_value=torch.Tensor(self._wiring.erev_initializer()),
        )

        # Checks whether to use the water balance equation to determine sensor_size
        # When using the water balance equation, sensor_size should include the original input features and the water balance output
        real_sensory_size = self.sensory_size
        if self._use_wb_equations:
            # sensory_sizeIt will not adjust automatically, so we need to consider the water balance output in advance
            # NOTE: Make sure this matches the arguments to the wiring.build() call
            real_sensory_size = self.sensory_size

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value((real_sensory_size, self.regular_state_size), "sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value((real_sensory_size, self.regular_state_size), "sensory_mu"),
        )
        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value((real_sensory_size, self.regular_state_size), "sensory_w"),
        )
        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev", init_value=torch.Tensor(self._wiring.sensory_erev_initializer()),
        )

        self._params["sparsity_mask"] = self.add_weight(
            "sparsity_mask",
            torch.Tensor(np.abs(self._wiring.adjacency_matrix)),
            requires_grad=False,
        )
        self._params["sensory_sparsity_mask"] = self.add_weight(
            "sensory_sparsity_mask",
            torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)),
            requires_grad=False,
        )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((real_sensory_size,)),  # 使用实际的sensory_size
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((real_sensory_size,)),  # 使用实际的sensory_size
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,)),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,)),
            )

    def _init_wb_parameters(self):
        """Initialize water balance equation parameters and remove parameters related to soil layer"""
        self.wb_params = {}

        # Karst aquifer-specific parameters
        self.wb_params["karst_storage"] = self.add_weight(
            name="karst_storage", init_value=torch.rand(1) * 50.0 + 100.0  # 100-150
        )
        self.wb_params["karst_release"] = self.add_weight(
            name="karst_release", init_value=torch.rand(1) * 0.2 + 0.1  # 0.1-0.3
        )
        self.wb_params["karst_infiltration"] = self.add_weight(
            name="karst_infiltration", init_value=torch.rand(1) * 0.4 + 0.3  # 0.3-0.7
        )

        # Groundwater extraction parameters
        self.wb_params["pumping_factor"] = self.add_weight(
            name="pumping_factor", init_value=torch.rand(1) * 0.5 + 0.5  # 0.5-1.0
        )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        # The state here now only contains the neuron state
        v_pre = state

        # For tensor quantization, ensure that the shape of elapsed_time is correct
        # If elapsed_time is a scalar, convert it to a tensor of appropriate shape
        if isinstance(elapsed_time, (int, float)):
            elapsed_time = torch.tensor([elapsed_time], device=inputs.device)

        # Transform elapsed_time into a shape compatible with state for batch processing
        batch_size = inputs.size(0)
        if elapsed_time.shape[0] != batch_size:
            elapsed_time = elapsed_time.expand(batch_size)

        # Pre-computed sensor neuron effects
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
                sensory_w_activation * self._params["sensory_sparsity_mask"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduced in dimension 1 (= source sensory neurons)
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # For each sample, use its corresponding elapsed_time
        # Change cm_t calculation to batch mode
        cm = self.make_positive_fn(self._params["cm"])
        # Extend cm so that each sample uses its own elapsed_time
        cm_expanded = cm.unsqueeze(0).expand(batch_size, -1)
        # Calculate cm_t for each sample
        elapsed_time_expanded = elapsed_time.unsqueeze(1)
        cm_t = cm_expanded / (elapsed_time_expanded / self._ode_unfolds)

        # Unroll multiple ODEs into one RNN step
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation = w_activation * self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # Reduce on dimension 1 (=source neurons)
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            gleak_expanded = gleak.unsqueeze(0).expand(batch_size, -1)
            vleak_expanded = self._params["vleak"].unsqueeze(0).expand(batch_size, -1)

            numerator = cm_t * v_pre + gleak_expanded * vleak_expanded + w_numerator
            denominator = cm_t + gleak_expanded + w_denominator

            # Avoid division by 0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _wb_model_solve(self, inputs, wb_states, batch_size=1):
        """Use vectorized water balance method to remove vegetation interception, capillary and gravity reservoirs"""
        device = inputs.device

        # Extracting the water balance state from the state vector
        S5 = wb_states[:, 0:1]  # Karst aquifer reservoir

        # Parsing Input
        P = inputs[:, 0:1]  # precipitation
        T = inputs[:, 1:2]  # temperature
        GWE = inputs[:, 2:3] if inputs.shape[1] > 2 else torch.ones((batch_size, 1), device=device) * 0.5  # Groundwater extraction

        # Simplified treatment: directly treating rainfall as surface runoff
        Qhor = P

        # Karst aquifer parameters
        karst_storage = self.wb_params["karst_storage"]  # Karst water storage capacity
        karst_release = self.wb_params["karst_release"]  # Release coefficient of karst reservoir

        # Calculating the amount of water entering the karst aquifer
        Qtokarst = Qhor * self.wb_params["karst_infiltration"]

        # Calculating the amount of water released from karst aquifers
        s5_ratio = S5 / karst_storage  # Karst aquifer saturation
        Qkarst_release = S5 * karst_release * torch.sigmoid(s5_ratio * 10)

        # Adding the impact of groundwater extraction
        pumping_factor = self.wb_params["pumping_factor"]  # Pumping coefficient
        actual_extraction = GWE * pumping_factor * torch.sigmoid(S5 / (karst_storage * 0.1))

        # Renewal of karst aquifer reservoirs
        new_S5 = S5 + Qtokarst - Qkarst_release - actual_extraction

        # Groundwater level changes are mainly determined by karst aquifers
        Qgw = Qtokarst - Qkarst_release - actual_extraction

        # Returns the groundwater level change and the new state
        return Qgw, new_S5

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        # If the water balance equation is used, state contains the neuron state and the water balance state.
        if self._use_wb_equations:
            # Only take part of the neuron state for output mapping
            output = state[:, :self.regular_state_size]
        else:
            output = state

        if self.motor_size < self.regular_state_size:
            output = output[:, 0: self.motor_size]  # slice

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            # In implicit mode, the parameter constraints are implemented via
            # a softplus function at runtime
            self._params["w"].data = self._clip(self._params["w"].data)
            self._params["sensory_w"].data = self._clip(self._params["sensory_w"].data)
            self._params["cm"].data = self._clip(self._params["cm"].data)
            self._params["gleak"].data = self._clip(self._params["gleak"].data)

    def forward(self, inputs, states, elapsed_time=1.0):
        """
        LTCCell's forward propagation method

        parameter:
        inputs: Input features, may already contain water balance output (if passed from LTC)
        states: Hidden state, including neuron state and water balance state
        elapsed_time: Time step

        (outputs, next_state)Tuple
        """
        # Get the batch size
        batch_size = inputs.size(0) if len(inputs.shape) > 1 else 1

        # Make sure elapsed_time is a tensor and has the correct shape
        if isinstance(elapsed_time, (int, float)):
            elapsed_time = torch.tensor([elapsed_time] * batch_size, device=inputs.device)
        elif len(elapsed_time.shape) == 0:  # A single scalar tensor
            elapsed_time = elapsed_time.expand(batch_size)

        # If the water balance equation is used
        if self._use_wb_equations:
            # Dissociating neuronal status and water balance status
            if len(states.shape) > 1:
                neuron_states = states[:, :self.regular_state_size]
                wb_states = states[:, self.regular_state_size:]
            else:
                neuron_states = states[:self.regular_state_size]
                wb_states = states[self.regular_state_size:]

            # Check the number of input features
            if batch_size > 1:
                feature_count = inputs.size(1)
            else:
                feature_count = len(inputs)

            # Determine whether the input already contains the water balance output
            # The input feature number should be the original feature number (6) + the horizontal balance output (1) = 7
            expected_feature_count = self.sensory_size

            if feature_count != expected_feature_count:
                raise RuntimeError(
                    f"Input dimension mismatch in LTCCell.forward: expected {expected_feature_count} "
                    f"features but got {feature_count}. This suggests a problem with water balance "
                    f"output handling."
                )

            # Note: At this point the inputs should already include the water balance output (passed from LTC forward)
            # So there is no need to calculate and connect the water balance output again
            # Only the water balance status needs to be updated

            # Only pass the first 4 features for water balance calculation and update the status
            wb_inputs = inputs[:, :4] if batch_size > 1 else inputs[:4]
            _, next_wb_state = self._wb_model_solve(wb_inputs, wb_states, batch_size)

            # Map the input (the input already includes the water balance output)
            mapped_inputs = self._map_inputs(inputs)

            # Apply the ODE solver to update the neuron state
            next_neuron_state = self._ode_solver(mapped_inputs, neuron_states, elapsed_time)

            # Merging neuronal status and water balance status
            next_state = torch.cat([next_neuron_state, next_wb_state], dim=1)
        else:
            # Mapping the input
            mapped_inputs = self._map_inputs(inputs)
            # Original LTC forward propagation
            next_state = self._ode_solver(mapped_inputs, states, elapsed_time)

        # Mapping Output
        outputs = self._map_outputs(next_state)

        return outputs, next_state
