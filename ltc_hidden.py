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

        # 确保在wiring.build()时使用正确的输入尺寸
        if not wiring.is_built():
            wiring.build(input_size)

        # 创建LTC单元，传递use_wb_equations参数
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

        # 如果使用混合记忆，确保LSTM接收正确维度的输入
        if self.use_mixed:
            # LSTM应接收与实际输入维度相同的输入，而不是总是假设有水平衡输出
            self.lstm = LSTMCell(input_size, self.state_size_without_wb())

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    def state_size_without_wb(self):
        """返回不包括水平衡状态的神经元状态大小"""
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
        前向传播方法

        参数:
        input: 输入张量，形状为(L,C)（无批次模式）或(B,L,C)（batch_first=True）或(L,B,C)（batch_first=False）
        hx: RNN的初始隐藏状态
        timespans: 可选的时间步长张量

        返回:
        (output, hx)元组，其中output为RNN输出，hx为最终隐藏状态
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

        # 初始化状态，根据是否使用水平衡方程进行不同处理
        if hx is None:
            if self.use_wb_equations:
                # 水平衡状态需要4个额外的状态变量
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

            # 根据是否使用水平衡方程进行不同处理
            if self.use_wb_equations:
                # 从隐藏状态中分离出神经元状态和水平衡状态
                neuron_state = h_state[:, :self.state_size_without_wb()]
                wb_state = h_state[:, self.state_size_without_wb():]

                # 只使用前4个特征计算水平衡
                wb_inputs = inputs_t[:, :4]

                # 计算水平衡输出并获取新的水平衡状态
                wb_output, new_wb_state = self.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)

                # 将水平衡输出附加到输入，创建增强输入
                augmented_inputs = torch.cat([inputs_t, wb_output], dim=1)

                # 更新h_state中的水平衡状态部分
                h_state_new = h_state.clone()
                h_state_new[:, self.state_size_without_wb():] = new_wb_state
                h_state = h_state_new
            else:
                # 不使用水平衡时，直接使用原始输入
                augmented_inputs = inputs_t

            # 混合记忆模式处理
            if self.use_mixed:
                if self.use_wb_equations:
                    # 提取神经元状态
                    neuron_state = h_state[:, :self.state_size_without_wb()]

                    # 使用LSTM处理，确保输入维度正确
                    h_lstm, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                    # 只更新h_state的神经元部分，保留水平衡状态
                    h_state_new = h_state.clone()
                    h_state_new[:, :self.state_size_without_wb()] = h_lstm
                    h_state = h_state_new
                else:
                    # 不使用水平衡时的LSTM更新
                    h_state, c_state = self.lstm(inputs_t, (h_state, c_state))

            # 前向传播LTC单元
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