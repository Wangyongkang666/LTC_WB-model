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
            use_wb_equations=True,  # 添加参数以控制是否使用水平衡方程
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
        self._use_wb_equations = use_wb_equations  # 是否使用水平衡方程
        self._allocate_parameters()

        # 水平衡方程参数
        if self._use_wb_equations:
            self._init_wb_parameters()

    @property
    def state_size(self):
        # 状态大小现在包括神经元状态和水平衡状态（如果启用）
        if self._use_wb_equations:
            return self._wiring.units + 1  # +4 用于水平衡状态(移除了积雪水库)
        else:
            return self._wiring.units

    @property
    def regular_state_size(self):
        # 原始状态大小（不包括水平衡状态）
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

        # 检查是否使用水平衡方程，以确定sensory_size
        # 在使用水平衡方程时，sensory_size应该包括原始输入特征和水平衡输出
        real_sensory_size = self.sensory_size
        if self._use_wb_equations:
            # sensory_size不会自动调整，因此我们需要预先考虑水平衡输出
            # 注意: 确保这与wiring.build()调用的参数一致
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
        """初始化水平衡方程参数,移除与土壤层相关的参数"""
        self.wb_params = {}

        # 岩溶含水层特有参数
        self.wb_params["karst_storage"] = self.add_weight(
            name="karst_storage", init_value=torch.rand(1) * 50.0 + 100.0  # 100-150
        )
        self.wb_params["karst_release"] = self.add_weight(
            name="karst_release", init_value=torch.rand(1) * 0.2 + 0.1  # 0.1-0.3
        )
        self.wb_params["karst_infiltration"] = self.add_weight(
            name="karst_infiltration", init_value=torch.rand(1) * 0.4 + 0.3  # 0.3-0.7
        )

        # 地下水开采参数
        self.wb_params["pumping_factor"] = self.add_weight(
            name="pumping_factor", init_value=torch.rand(1) * 0.5 + 0.5  # 0.5-1.0
        )

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        # 这里的state现在只包含神经元状态
        v_pre = state

        # 为张量化处理，确保elapsed_time形状正确
        # 如果elapsed_time是标量，转换为适当形状的张量
        if isinstance(elapsed_time, (int, float)):
            elapsed_time = torch.tensor([elapsed_time], device=inputs.device)

        # 将elapsed_time变成与state兼容的形状以便于批处理计算
        batch_size = inputs.size(0)
        if elapsed_time.shape[0] != batch_size:
            elapsed_time = elapsed_time.expand(batch_size)

        # 预计算传感神经元的影响
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = (
                sensory_w_activation * self._params["sensory_sparsity_mask"]
        )

        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # 在维度1上减少（=源感觉神经元）
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # 对每个样本使用其对应的elapsed_time
        # 将cm_t计算改为批处理方式
        cm = self.make_positive_fn(self._params["cm"])
        # 扩展cm以便于每个样本使用自己的elapsed_time
        cm_expanded = cm.unsqueeze(0).expand(batch_size, -1)
        # 计算每个样本的cm_t
        elapsed_time_expanded = elapsed_time.unsqueeze(1)
        cm_t = cm_expanded / (elapsed_time_expanded / self._ode_unfolds)

        # 展开多个ODE到一个RNN步骤
        w_param = self.make_positive_fn(self._params["w"])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            w_activation = w_activation * self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]

            # 在维度1上减少（=源神经元）
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            gleak = self.make_positive_fn(self._params["gleak"])
            gleak_expanded = gleak.unsqueeze(0).expand(batch_size, -1)
            vleak_expanded = self._params["vleak"].unsqueeze(0).expand(batch_size, -1)

            numerator = cm_t * v_pre + gleak_expanded * vleak_expanded + w_numerator
            denominator = cm_t + gleak_expanded + w_denominator

            # 避免除以0
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _wb_model_solve(self, inputs, wb_states, batch_size=1):
        """使用向量化计算的水平衡方程方法,移除植被截留、毛细管和重力水库"""
        device = inputs.device

        # 从状态向量中提取水平衡状态
        S5 = wb_states[:, 0:1]  # 岩溶含水层水库

        # 解析输入
        P = inputs[:, 0:1]  # 降水
        T = inputs[:, 1:2]  # 温度
        GWE = inputs[:, 2:3] if inputs.shape[1] > 2 else torch.ones((batch_size, 1), device=device) * 0.5  # 地下水开采

        # 简化处理：直接将降雨视为地表径流
        Qhor = P

        # 岩溶含水层参数
        karst_storage = self.wb_params["karst_storage"]  # 岩溶储水能力
        karst_release = self.wb_params["karst_release"]  # 岩溶水库释放系数

        # 计算进入岩溶含水层的水量
        Qtokarst = Qhor * self.wb_params["karst_infiltration"]

        # 计算从岩溶含水层释放的水量
        s5_ratio = S5 / karst_storage  # 岩溶含水层饱和度
        Qkarst_release = S5 * karst_release * torch.sigmoid(s5_ratio * 10)

        # 加入地下水开采的影响
        pumping_factor = self.wb_params["pumping_factor"]  # 抽水系数
        actual_extraction = GWE * pumping_factor * torch.sigmoid(S5 / (karst_storage * 0.1))

        # 更新岩溶含水层水库
        new_S5 = S5 + Qtokarst - Qkarst_release - actual_extraction

        # 地下水位变化主要由岩溶含水层决定
        Qgw = Qtokarst - Qkarst_release - actual_extraction

        # 返回地下水位变化和新的状态
        return Qgw, new_S5

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        # 如果使用水平衡方程，state包含神经元状态和水平衡状态
        if self._use_wb_equations:
            # 只取神经元状态部分进行输出映射
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
        LTCCell的前向传播方法

        参数:
        inputs: 输入特征，可能已经包含水平衡输出（如果从LTC传递）
        states: 隐藏状态，包含神经元状态和水平衡状态
        elapsed_time: 时间步长

        返回:
        (outputs, next_state)元组
        """
        # 获取批大小
        batch_size = inputs.size(0) if len(inputs.shape) > 1 else 1

        # 确保 elapsed_time 是张量并且形状正确
        if isinstance(elapsed_time, (int, float)):
            elapsed_time = torch.tensor([elapsed_time] * batch_size, device=inputs.device)
        elif len(elapsed_time.shape) == 0:  # 单个标量张量
            elapsed_time = elapsed_time.expand(batch_size)

        # 如果使用水平衡方程
        if self._use_wb_equations:
            # 分离神经元状态和水平衡状态
            if len(states.shape) > 1:
                neuron_states = states[:, :self.regular_state_size]
                wb_states = states[:, self.regular_state_size:]
            else:
                neuron_states = states[:self.regular_state_size]
                wb_states = states[self.regular_state_size:]

            # 检查输入特征的数量
            if batch_size > 1:
                feature_count = inputs.size(1)
            else:
                feature_count = len(inputs)

            # 判断输入是否已经包含水平衡输出
            # 输入特征数应该是原始特征数(6) + 水平衡输出(1) = 7
            expected_feature_count = self.sensory_size  # 应该是7

            if feature_count != expected_feature_count:
                raise RuntimeError(
                    f"Input dimension mismatch in LTCCell.forward: expected {expected_feature_count} "
                    f"features but got {feature_count}. This suggests a problem with water balance "
                    f"output handling."
                )

            # 注意：此时的inputs应该已经包含了水平衡输出（从LTC的forward传递）
            # 所以不需要再次计算和连接水平衡输出
            # 只需要更新水平衡状态

            # 只传递前4个特征进行水平衡计算，更新状态
            wb_inputs = inputs[:, :4] if batch_size > 1 else inputs[:4]
            _, next_wb_state = self._wb_model_solve(wb_inputs, wb_states, batch_size)

            # 对输入进行映射（此时输入已经包含水平衡输出）
            mapped_inputs = self._map_inputs(inputs)

            # 应用ODE求解器更新神经元状态
            next_neuron_state = self._ode_solver(mapped_inputs, neuron_states, elapsed_time)

            # 合并神经元状态和水平衡状态
            next_state = torch.cat([next_neuron_state, next_wb_state], dim=1)
        else:
            # 对输入进行映射
            mapped_inputs = self._map_inputs(inputs)
            # 原始LTC前向传播
            next_state = self._ode_solver(mapped_inputs, states, elapsed_time)

        # 映射输出
        outputs = self._map_outputs(next_state)

        return outputs, next_state