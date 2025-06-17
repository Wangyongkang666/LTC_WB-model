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
            return self._wiring.units + 5  # +4 用于水平衡状态(移除了积雪水库)
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
        """初始化水平衡方程参数，添加第四纪覆盖层和岩溶含水层之间的界面水库"""
        self.wb_params = {}

        # 冠层拦截参数
        self.wb_params["SCmax_summer"] = self.add_weight(
            name="SCmax_summer", init_value=torch.rand(1) * 0.05 + 0.01  # 0.01-0.06
        )
        self.wb_params["SCmax_winter"] = self.add_weight(
            name="SCmax_winter", init_value=torch.rand(1) * 0.02 + 0.01  # 0.01-0.03
        )
        self.wb_params["Kc_summer"] = self.add_weight(
            name="Kc_summer", init_value=torch.rand(1) * 0.1 + 0.1  # 0.1-0.2
        )
        self.wb_params["Kc_winter"] = self.add_weight(
            name="Kc_winter", init_value=torch.rand(1) * 0.05 + 0.05  # 0.05-0.1
        )

        # 土壤渗透参数 - 第四纪覆盖层
        self.wb_params["Rin"] = self.add_weight(
            name="Rin", init_value=torch.rand(1) * 0.2 + 0.6  # 0.6-0.8
        )
        self.wb_params["Rpr"] = self.add_weight(
            name="Rpr", init_value=torch.rand(1) * 0.2 + 0.2  # 0.2-0.4
        )
        self.wb_params["f"] = self.add_weight(
            name="f", init_value=torch.rand(1) * 0.1 + 0.3  # 0.3-0.4
        )

        # 第四纪覆盖层储水容量参数
        self.wb_params["Spmax"] = self.add_weight(
            name="Spmax", init_value=torch.rand(1) * 20.0 + 15.0  # 15-35
        )
        self.wb_params["Qpmax"] = self.add_weight(
            name="Qpmax", init_value=torch.rand(1) * 15.0 + 10.0  # 10-25
        )
        self.wb_params["Scmax"] = self.add_weight(
            name="Scmax", init_value=torch.rand(1) * 20.0 + 20.0  # 20-40
        )
        self.wb_params["Scmin"] = self.add_weight(
            name="Scmin", init_value=torch.rand(1) * 5.0 + 2.0  # 2-7
        )

        # 第四纪覆盖层流动参数
        self.wb_params["kl"] = self.add_weight(
            name="kl", init_value=torch.rand(1) * 0.1 + 0.05  # 0.05-0.15
        )
        self.wb_params["kn"] = self.add_weight(
            name="kn", init_value=torch.rand(1) * 0.1 + 0.05  # 0.05-0.15
        )
        self.wb_params["Sgmax"] = self.add_weight(
            name="Sgmax", init_value=torch.rand(1) * 30.0 + 40.0  # 40-70
        )
        self.wb_params["Qgmax"] = self.add_weight(
            name="Qgmax", init_value=torch.rand(1) * 15.0 + 10.0  # 10-25
        )

        # 岩溶含水层特有参数
        self.wb_params["karst_connect"] = self.add_weight(
            name="karst_connect", init_value=torch.rand(1) * 0.3 + 0.3  # 0.3-0.6
        )
        self.wb_params["karst_storage"] = self.add_weight(
            name="karst_storage", init_value=torch.rand(1) * 50.0 + 100.0  # 100-150
        )

        # 新增：覆盖层-岩溶界面参数
        self.wb_params["interface_capacity"] = self.add_weight(
            name="interface_capacity", init_value=torch.rand(1) * 30.0 + 20.0  # 20-50
        )
        self.wb_params["interface_release"] = self.add_weight(
            name="interface_release", init_value=torch.rand(1) * 0.2 + 0.1  # 0.1-0.3
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
        """使用向量化计算的水平衡方程方法，添加覆盖层-岩溶界面水库"""
        device = inputs.device

        # 从状态向量中提取水平衡状态 - 添加岩溶水库
        S0 = wb_states[:, 0:1]  # 冠层拦截水库
        S2 = wb_states[:, 1:2]  # 优先流水库（第四纪覆盖层）
        S3 = wb_states[:, 2:3]  # 毛细管水库（第四纪覆盖层）
        S4 = wb_states[:, 3:4]  # 重力水库（第四纪覆盖层）
        S5 = wb_states[:, 4:5]  # 新增：岩溶含水层水库

        # 解析输入
        P = inputs[:, 0:1]  # 降水
        T = inputs[:, 1:2]  # 温度
        DayL = inputs[:, 2:3]  # 日长
        GWE = inputs[:, 3:4] if inputs.shape[1] > 3 else torch.ones((batch_size, 1), device=device) * 0.5  # 地下水开采

        # 计算PET (潜在蒸散发) - 向量化
        PET = 29.8 * DayL * 0.611 * torch.exp(17.3 * T / (T + 237.3)) / (T + 237.3)

        # 创建季节掩码
        summer_mask = (DayL >= 0.5)

        # 1. 冠层拦截 - 向量化季节参数选择
        SCmax = torch.where(summer_mask, self.wb_params["SCmax_summer"], self.wb_params["SCmax_winter"])
        Kc = torch.where(summer_mask, self.wb_params["Kc_summer"], self.wb_params["Kc_winter"])

        # 简化版的冠层拦截计算 - 向量化条件判断
        Dc = 0.3  # 植被覆盖率减小为0.3
        A = 1.0

        # 计算拦截量 - 向量化条件判断
        s0_lt_0 = (S0 < 0)
        s0_between_0_SCmax = (S0 >= 0) & (S0 <= SCmax)
        s0_gt_SCmax = (S0 > SCmax)

        Pint = torch.zeros_like(P, device=device)
        Pint = torch.where(s0_between_0_SCmax, Kc * Dc * A, Pint)
        Pint = torch.where(s0_gt_SCmax, P, Pint)

        # 更新冠层拦截水库
        new_S0 = S0 + Pint

        # 2. 计算直达地面的降水 - 向量化
        P_ground = P - Pint

        # 简化处理：移除积雪过程，直接使用地表降水
        Pr = P_ground  # 全部作为雨水处理

        # 4. 第四纪覆盖层土壤水分渗透过程 - 向量化参数
        Rin = self.wb_params["Rin"]  # 渗透率
        Rpr = self.wb_params["Rpr"]  # 优先流系数
        Spmax = self.wb_params["Spmax"]  # 优先流水库容量
        Qpmax = self.wb_params["Qpmax"]  # 最大优先流
        f = self.wb_params["f"]  # 衰减因子

        # 霍顿地表径流 - 向量化
        Qhor = Pr * (1 - Rin)

        # 优先流 - 向量化条件判断（不再使用karst_connect参数）
        s2_lt_0 = (S2 < 0)
        s2_between_0_Spmax = (S2 >= 0) & (S2 <= Spmax)
        s2_gt_Spmax = (S2 > Spmax)

        Qpref = torch.zeros_like(P, device=device)
        Qpref = torch.where(s2_between_0_Spmax,
                            Pr * Rin * Rpr * torch.exp(-f * (Spmax - S2)),
                            Qpref)
        Qpref = torch.where(s2_gt_Spmax, Qpmax, Qpref)

        # 更新优先流水库
        new_S2 = S2 + Pr * Rin * Rpr

        # 5. 毛细管水库 - 向量化参数和条件判断
        Scmax = self.wb_params["Scmax"]  # 毛细管水库容量
        Scmin = self.wb_params["Scmin"]

        # 进入毛细管水库的水量
        Qcap = Pr * Rin * (1 - Rpr)

        # 计算实际蒸散发 - 向量化条件判断
        s3_lt_Scmin = (S3 < Scmin)
        s3_between_Scmin_Scmax = (S3 >= Scmin) & (S3 <= Scmax)
        s3_gt_Scmax = (S3 > Scmax)

        ET = torch.zeros_like(P, device=device)
        ET = torch.where(s3_between_Scmin_Scmax, PET * (S3 / Scmax), ET)
        ET = torch.where(s3_gt_Scmax, PET, ET)

        # 更新毛细管水库
        new_S3 = S3 + Qcap - ET

        # 6. 重力水库 (第四纪覆盖层底部) - 向量化参数和条件判断
        kl = self.wb_params["kl"]  # 线性系数
        kn = self.wb_params["kn"]  # 非线性系数
        Sgmax = self.wb_params["Sgmax"]
        Qgmax = self.wb_params["Qgmax"]  # 最大流量

        # 计算进入重力水库的水量
        Qgra = Qcap - ET

        # 计算慢流 - 向量化条件判断 (这里的慢流将流向岩溶水库)
        s4_lt_0 = (S4 < 0)
        s4_between_0_Sgmax = (S4 >= 0) & (S4 <= Sgmax)
        s4_gt_Sgmax = (S4 > Sgmax)

        # 慢流水量
        Qslow = torch.zeros_like(P, device=device)
        Qslow = torch.where(s4_between_0_Sgmax,
                            Qgra * kl + Qgra * Qgra * kn * torch.exp(-f * (Sgmax - S4)),
                            Qslow)
        Qslow = torch.where(s4_gt_Sgmax, Qgmax, Qslow)

        # 更新重力水库（第四纪覆盖层）
        new_S4 = S4 + Qgra - Qslow

        # 7. 新增：岩溶含水层处理
        karst_connect = self.wb_params["karst_connect"]  # 岩溶连通度
        karst_storage = self.wb_params["karst_storage"]  # 岩溶储水能力
        interface_capacity = self.wb_params["interface_capacity"]  # 界面容量
        interface_release = self.wb_params["interface_release"]  # 界面释放率
        karst_infiltration = self.wb_params["karst_infiltration"]  # 岩溶渗透率

        # 计算从覆盖层流向岩溶含水层的水量
        # 现在，优先流和慢流都会进入岩溶含水层，受岩溶连通性影响
        Qtokarst = (Qpref + Qslow) * karst_infiltration

        # 计算从岩溶含水层释放的水量，受岩溶连通性影响
        s5_ratio = S5 / karst_storage  # 岩溶含水层饱和度
        Qkarst_release = S5 * interface_release * karst_connect * torch.sigmoid(s5_ratio * 10)

        # 7. 加入地下水开采的影响 - 向量化，现在从岩溶含水层抽水
        pumping_factor = self.wb_params["pumping_factor"]  # 抽水系数
        actual_extraction = GWE * pumping_factor * torch.sigmoid(S5 / (karst_storage * 0.1))

        # 更新岩溶含水层水库
        new_S5 = S5 + Qtokarst - Qkarst_release - actual_extraction

        # 8. 最终输出 - 向量化
        # 地下水位变化主要由岩溶含水层决定
        Qgw = Qtokarst - Qkarst_release - actual_extraction

        # 将所有更新后的状态合并为一个张量（包含新增的岩溶含水层）
        new_wb_states = torch.cat([new_S0, new_S2, new_S3, new_S4, new_S5], dim=1)

        return Qgw, new_wb_states

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
                # 如果inputs被错误地展平为一维，重新塑造它
                inputs = inputs.reshape(1, -1)
                feature_count = inputs.size(1)

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