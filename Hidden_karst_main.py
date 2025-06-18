import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from datetime import datetime
import math
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import networkx as nx
from scipy.stats import pearsonr
import time
# 设置全局样式
sns.set(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "DejaVu Sans"


###########################
# 数据加载与预处理
###########################

def load_trace():
    # 假设数据文件位于当前目录的data/traffic目录下
    df = pd.read_csv("D:\研一下\LNNS预测\LTC_多变量试验\日尺度\well_250364\site_250364.csv")

    time_intervals = df["time_interval"].values.astype(np.float32) / 5.0
    # 处理特征
    pre = df["pre"].values.astype(np.float32)
    pre -= np.mean(pre)
    pre /= np.std(pre)

    # 添加温度数据
    temperature = df["temperature"].values.astype(np.float32)
    temperature -= np.mean(temperature)
    temperature /= np.std(temperature)

    # 时间特征处理
    date_time = [datetime.strptime(d, "%Y/%m/%d %H:%M") for d in df["date_time"]]
    day_of_year = [d.timetuple().tm_yday for d in date_time]
    day_length = []
    for doy in day_of_year:
        # 偏移日期使冬至（约第355天）对应正弦波最小值
        shifted_day = (doy - 83) % 365  # 或者更精确的偏移量
        angle = (2 * math.pi * shifted_day) / 365
        relative_length = (math.sin(angle) + 1) / 2
        day_length.append(relative_length)

    # Convert to numpy array with float32 type
    day_length = np.array(day_length).astype(np.float32)

    # 添加地下水开采数据，如果没有，可以基于季节性或其他因素构建模拟值
    if "ground_ext" in df.columns:
        extraction = df["ext"].values.astype(np.float32)
        extraction -= np.mean(extraction)
        extraction /= np.std(extraction)
    else:
        # 模拟值：夏季开采量大，冬季开采量小
        extraction = np.array([0.7 if 5 <= d.month <= 9 else 0.3 for d in date_time]).astype(np.float32)
        extraction -= np.mean(extraction)
        extraction /= np.std(extraction)
        # 添加两个新特征（举例）
    if "srv" in df.columns:
        feature5 = df["srv"].values.astype(np.float32)
        feature5 -= np.mean(feature5)
        feature5 /= np.std(feature5)
    else:
        # 如果没有该特征，可以用模拟数据或派生特征
        feature5 = np.zeros_like(pre)  # 或其他合适的初始化方式

    if "sp" in df.columns:
        feature6 = df["sp"].values.astype(np.float32)
        feature6 -= np.mean(feature6)
        feature6 /= np.std(feature6)
    else:
        feature6 = np.zeros_like(pre)  # 或其他合适的初始化方式

        # 组合特征
    features = np.stack([pre, temperature, day_length, extraction, feature5, feature6], axis=-1)

    # 处理目标变量
    well_level = df["well_level"].values.astype(np.float32).reshape(-1, 1)

    return features, well_level, time_intervals


def create_sequences(features, target, time_intervals, seq_len=12, inc=1):
    sequences_x = []
    sequences_y = []
    sequences_time = []  # 用于存储每个序列最后一步的时间间隔

    for s in range(0, len(features) - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(features[start:end])
        sequences_y.append(target[end - 1])
        sequences_time.append(time_intervals[start:end])  # 添加对应的时间间隔序列

    return (np.stack(sequences_x, axis=0),
            np.array(sequences_y).reshape(-1, 1),
            np.stack(sequences_time, axis=0))  # 返回时间间隔序列


# 数据模块
###########################

class TrafficDataModule(pl.LightningDataModule):
    def __init__(self, seq_len=36, batch_size=64):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def prepare_data(self):
        features, target, time_intervals = load_trace()

        target = self.scaler_y.fit_transform(target)

        train_x, train_y, train_times = create_sequences(features, target, time_intervals, self.seq_len, inc=1)

        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        self.train_times = np.stack(train_times, axis=0)  # 存储时间间隔
        total_seqs = self.train_x.shape[0]
        print("Total number of training sequences: {}".format(total_seqs))
        valid_size = int(0.15 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.test_x = self.train_x[-test_size:]
        self.test_y = self.train_y[-test_size:]
        self.test_times = self.train_times[-test_size:]  # 添加测试集时间间隔
        self.valid_x = self.train_x[-(test_size + valid_size):-test_size]
        self.valid_y = self.train_y[-(test_size + valid_size):-test_size]
        self.valid_times = self.train_times[-(test_size + valid_size):-test_size]  # 添加验证集时间间隔
        self.train_x = self.train_x[:-(test_size + valid_size)]
        self.train_y = self.train_y[:-(test_size + valid_size)]
        self.train_times = self.train_times[:-(test_size + valid_size)]

    def train_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.train_x, dtype=torch.float32),
            torch.tensor(self.train_y, dtype=torch.float32),
            torch.tensor(self.train_times, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.valid_x, dtype=torch.float32),
            torch.tensor(self.valid_y, dtype=torch.float32),
            torch.tensor(self.valid_times, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = TensorDataset(
            torch.tensor(self.test_x, dtype=torch.float32),
            torch.tensor(self.test_y, dtype=torch.float32),
            torch.tensor(self.test_times, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=self.batch_size)


###########################
# LTC模型
###########################

class LTCForecaster(pl.LightningModule):
    def __init__(self, input_size=6, ncp_units=32, lr=0.01, use_wb_equations=True, mixed_memory=True):
        """
        初始化LTC预测模型

        参数:
        input_size: 原始输入特征的数量
        ncp_units: 神经网络单元数量
        lr: 学习率
        use_wb_equations: 是否使用水平衡方程
        """
        super().__init__()
        self.lr = lr
        self.input_size = input_size
        self.use_wb_equations = use_wb_equations
        self.mixed_memory = mixed_memory  # 添加mixed_memory属性

        # 创建神经元接线配置
        self.wiring = AutoNCP(ncp_units, 1, sparsity_level=0.5)

        # 为wiring.build()计算正确的输入维度
        extended_input_size = input_size + 1 if use_wb_equations else input_size

        # 使用扩展后的输入维度构建wiring
        self.wiring.build(extended_input_size)

        if self.mixed_memory:
            self.lstm = nn.LSTMCell(
                input_size=extended_input_size,
                hidden_size=self.wiring.units
            )
        # 为LTC准备输入尺寸
        self.ltc = LTC(
            input_size=extended_input_size,
            units=self.wiring,
            return_sequences=False,
            batch_first=True,
            mixed_memory=self.mixed_memory,
            use_wb_equations=use_wb_equations
        )
        # 用于跟踪训练和验证损失
        self.train_loss = []
        self.val_loss = []

    def forward(self, x, time_intervals=None):
        # 获取批次大小
        batch_size = x.size(0)
        seq_len = x.size(1)
        # 计算神经状态大小
        neural_state_size = self.wiring.units

        # 根据是否使用水平衡方程初始化不同的状态
        if self.use_wb_equations:
            # 水平衡状态需要4个额外的状态变量
            total_state_size = neural_state_size + 5
            h_state = torch.zeros(batch_size, total_state_size, device=x.device)
        else:
            h_state = torch.zeros(batch_size, neural_state_size, device=x.device)

        # 创建LSTM单元状态 c0 (只针对神经元状态大小)
        c_state = torch.zeros(batch_size, neural_state_size, device=x.device)

        # 将它们组合为元组
        initial_state = (h_state, c_state)

        # 使用存储所有时间步的输出
        outputs = []

        # 当前状态初始化为初始状态
        current_state = initial_state

        # 逐时间步处理序列
        for t in range(seq_len):
            # 获取当前时间步的输入
            x_t = x[:, t, :]
            if len(x_t.shape) == 1:
                x_t = x_t.unsqueeze(0)
            # 获取当前时间步的时间间隔
            if time_intervals is not None:
                elapsed_time = time_intervals[:, t]
            else:
                elapsed_time = torch.ones(batch_size, device=x.device)  # 默认为1.0

            # 处理水平衡方程
            if self.use_wb_equations:
                h_state, c_state = current_state

                # 从隐藏状态中分离出神经元状态和水平衡状态
                neuron_state = h_state[:, :neural_state_size]
                wb_state = h_state[:, neural_state_size:]

                # 只使用前4个特征计算水平衡
                wb_inputs = x_t[:, :4]
                # 计算水平衡输出并获取新的水平衡状态
                wb_output, new_wb_state = self.ltc.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)
                # 将水平衡输出附加到输入，创建增强输入
                augmented_inputs = torch.cat([x_t, wb_output], dim=1)

                # 对于LSTM用于混合记忆模式，使用原始神经元状态
                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                # 前向传播LTC单元，使用当前时间间隔
                output, new_h_state = self.ltc.rnn_cell.forward(
                    augmented_inputs,
                    torch.cat([neuron_state, wb_state], dim=1),
                    elapsed_time=elapsed_time
                )

                # 更新h_state中的水平衡状态部分（根据实际返回结构调整）
                if self.use_wb_equations:
                    # 假设new_h_state已经是整合后的状态
                    new_h_state = torch.cat([
                        new_h_state[:, :neural_state_size],
                        new_wb_state
                    ], dim=1)

                # 更新当前状态
                current_state = (new_h_state, c_state)
            else:
                # 原始LTC处理，使用当前时间间隔
                h_state, c_state = current_state

                # 对于LSTM混合记忆模式
                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(x_t, (h_state, c_state))
                    h_state = neuron_state

                # 使用LTC的rnn_cell而不是self.rnn_cell
                output, new_h_state = self.ltc.rnn_cell.forward(x_t, h_state, elapsed_time=elapsed_time)

                # 更新当前状态
                current_state = (new_h_state, c_state)

            outputs.append(output)

        # 返回最后一个时间步的输出或整个序列
        if self.ltc.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return outputs[-1]

    def training_step(self, batch, batch_idx):
        x, y, time_intervals = batch  # 更新以处理时间间隔
        y_hat = self(x, time_intervals)  # 传递时间间隔
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_loss.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, time_intervals = batch  # 更新以处理时间间隔
        y_hat = self(x, time_intervals)  # 传递时间间隔
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_loss.append(loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


# 添加到原始代码的model类中
class LTCForecasterWithVisualization(LTCForecaster):
    """
    扩展LTC预测模型，添加可视化能力
    """

    def __init__(self, input_size=6, ncp_units=32, lr=0.01, use_wb_equations=True, mixed_memory=True):
        super().__init__(input_size, ncp_units, lr, use_wb_equations, mixed_memory)
        # 记录训练过程中的额外信息
        self.feature_importance = {}
        self.neuron_activations = []
        self.water_balance_states = []
        self.attention_weights = []

    def capture_neuron_activations(self, x, time_intervals=None):
        """捕获神经元激活值和水平衡状态"""
        # 获取批次大小
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化状态
        if self.use_wb_equations:
            total_state_size = self.wiring.units + 5
            h_state = torch.zeros(batch_size, total_state_size, device=x.device)
        else:
            h_state = torch.zeros(batch_size, self.wiring.units, device=x.device)

        c_state = torch.zeros(batch_size, self.wiring.units, device=x.device)

        initial_state = (h_state, c_state)
        current_state = initial_state

        # 记录每个时间步的神经元激活和水平衡状态
        activations = []
        wb_states = None  # 如果不使用水平衡，初始化为None
        if self.use_wb_equations:
            wb_states = []
        attn_weights = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            elapsed_time = time_intervals[:, t] if time_intervals is not None else torch.ones(batch_size,device=x.device)

            if self.use_wb_equations:
                h_state, c_state = current_state
                neuron_state = h_state[:, :self.wiring.units]
                wb_state = h_state[:, self.wiring.units:]

                # 计算水平衡
                wb_inputs = x_t[:, :4]
                wb_output, new_wb_state = self.ltc.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)

                # 保存水平衡状态
                wb_states.append(new_wb_state.detach().cpu().numpy())

                # 创建增强输入
                augmented_inputs = torch.cat([x_t, wb_output], dim=1)

                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                # 计算注意力权重（近似为输入与神经元状态的相关性）
                # 我们不能直接做矩阵乘法，而应该计算每个特征的重要性
                attention = torch.zeros((batch_size, augmented_inputs.size(1)), device=x.device)

                # 简单方法：使用特征与第一个输出神经元的连接强度作为重要性
                for i in range(augmented_inputs.size(1)):
                    # 对于每个输入特征，计算它与神经元状态的相关性
                    corr = torch.mean(augmented_inputs[:, i].unsqueeze(1) * neuron_state, dim=1)
                    attention[:, i] = corr
                attention = torch.softmax(attention, dim=1)
                attn_weights.append(attention.detach().cpu().numpy())

                # 前向传播
                output, new_h_state = self.ltc.rnn_cell.forward(
                    augmented_inputs,
                    torch.cat([neuron_state, wb_state], dim=1),
                    elapsed_time=elapsed_time
                )

                # 保存神经元激活
                activations.append(new_h_state[:, :self.wiring.units].detach().cpu().numpy())

                # 更新状态
                new_h_state = torch.cat([
                    new_h_state[:, :self.wiring.units],
                    new_wb_state
                ], dim=1)

                current_state = (new_h_state, c_state)
            else:
                # 不使用水平衡方程的情况
                h_state, c_state = current_state

                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(x_t, (h_state, c_state))
                    h_state = neuron_state

                attention = torch.zeros((batch_size, x_t.size(1)), device=x.device)
                for i in range(x_t.size(1)):
                    corr = torch.mean(x_t[:, i].unsqueeze(1) * h_state, dim=1)
                    attention[:, i] = corr
                attention = torch.softmax(attention, dim=1)

                attn_weights.append(attention.detach().cpu().numpy())

                output, new_h_state = self.ltc.rnn_cell.forward(x_t, h_state, elapsed_time=elapsed_time)

                activations.append(new_h_state.detach().cpu().numpy())
                current_state = (new_h_state, c_state)

        return np.array(activations), wb_states if self.use_wb_equations else None, np.array(attn_weights)

    def compute_feature_importance(self, dataloader):
        """计算特征重要性"""
        feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']
        importances = {name: [] for name in feature_names}

        with torch.no_grad():
            for x, y, times in dataloader:
                # 计算初始预测
                original_pred = self(x, times)

                # 对每个特征进行扰动并计算影响
                for i, feature_name in enumerate(feature_names):
                    # 创建扰动数据
                    perturbed_x = x.clone()
                    perturbed_x[:, :, i] = torch.mean(perturbed_x[:, :, i])

                    # 使用扰动数据预测
                    perturbed_pred = self(perturbed_x, times)

                    # 计算影响（使用MSE变化）
                    impact = torch.mean(torch.abs(original_pred - perturbed_pred)).item()
                    importances[feature_name].append(impact)

        # 计算平均影响
        for feature_name in feature_names:
            self.feature_importance[feature_name] = np.mean(importances[feature_name])

        # 归一化
        total = sum(self.feature_importance.values())
        for feature_name in feature_names:
            self.feature_importance[feature_name] /= total

        return self.feature_importance

###########################
# 可视化函数
###########################
# 可视化函数
def visualize_feature_importance(model, output_dir):
    """可视化特征重要性"""
    if not hasattr(model, 'feature_importance') or not model.feature_importance:
        print("模型没有计算特征重要性，无法可视化")
        return

    plt.figure(figsize=(10, 6))
    features = list(model.feature_importance.keys())
    importances = list(model.feature_importance.values())

    # 创建水平条形图
    bars = plt.barh(features, importances, color=plt.cm.viridis(np.linspace(0, 0.8, len(features))))

    # 添加数值标签
    for i, v in enumerate(importances):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.title('Feature importance analysis', fontsize=14)
    plt.xlabel('Importance Score (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Feature importance analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_neuron_dynamics(activations, wiring, output_dir, step=1):
    """可视化神经元动态"""
    if activations is None or len(activations) == 0:
        print("没有神经元激活数据，无法可视化")
        return

    # 选择部分时间步进行可视化
    time_steps = list(range(0, len(activations), step))
    if time_steps[-1] != len(activations) - 1:
        time_steps.append(len(activations) - 1)

    plt.figure(figsize=(15, 10))

    # 计算平均激活
    mean_activations = np.mean(activations, axis=1)

    # 为每个神经元创建一条时间序列线
    for i in range(min(wiring.units, 10)):  # 限制为前10个神经元，避免过度拥挤
        plt.plot(mean_activations[:, i], label=f'neuron {i + 1}', alpha=0.7)

    # 标记关键时间步
    for t in time_steps:
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)

    plt.title('Neuronal activity varies over time.', fontsize=14)
    plt.xlabel('time step')
    plt.ylabel('Average activation value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Neuronal dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 可视化神经元激活热图
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(mean_activations.T, cmap='viridis',
                     xticklabels=10, yticklabels=list(range(1, min(wiring.units, 100) + 1)))
    plt.title('Neuronal activation heat map', fontsize=14)
    plt.xlabel('Time step')
    plt.ylabel('Neuron number')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Neuronal activation heat map.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_water_balance(wb_states, time_intervals, input_data, output_dir):
    """可视化水平衡方程状态变化"""
    if wb_states is None or len(wb_states) == 0:
        print("没有水平衡状态数据，无法可视化")
        return

    # 水平衡组分名称
    wb_components = ['Canopy intercepts reservoirs.', 'Priority flow reservoirs', 'Capillary reservoirs', 'Gravity Reservoir','karset Reservoir']

    # 计算平均状态
    mean_wb_states = np.mean(wb_states, axis=1)

    # 绘制水平衡组分变化
    plt.figure(figsize=(14, 8))

    for i, component in enumerate(wb_components):
        plt.plot(mean_wb_states[:, i], label=component,
                 linewidth=2, marker='o', markersize=4, markevery=10)

    plt.title('The state of each component of the water balance changes over time.', fontsize=14)
    plt.xlabel('Time step')
    plt.ylabel('Status value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Changes in the state of water equilibrium.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制水平衡组分与输入特征的关系
    # 处理输入数据，获取降水和温度
    mean_inputs = np.mean(input_data, axis=0)
    precipitation = mean_inputs[:, 0]  # 假设降水是第一个特征
    temperature = mean_inputs[:, 1]  # 假设温度是第二个特征

    # 绘制多面板图表
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2])

    # 降水图
    ax1 = plt.subplot(gs[0])
    ax1.bar(range(len(precipitation)), precipitation, color='skyblue', alpha=0.7)
    ax1.set_title('Precipitation', fontsize=12)
    ax1.set_ylabel('Precipitation (Normalised)')
    ax1.grid(True, alpha=0.3)

    # 温度图
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(temperature, color='red', linewidth=2)
    ax2.set_title('Temperature', fontsize=12)
    ax2.set_ylabel('Temperature (Normalised)')
    ax2.grid(True, alpha=0.3)

    # 水平衡状态图
    ax3 = plt.subplot(gs[2], sharex=ax1)
    for i, component in enumerate(wb_components):
        ax3.plot(mean_wb_states[:, i], label=component, linewidth=2)

    ax3.set_title('Water balance component state', fontsize=12)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Status value')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Relationship between water balance and meteorological factors.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_attention_weights(attention_weights, feature_names, output_dir):
    """可视化注意力权重"""
    if attention_weights is None or len(attention_weights) == 0:
        print("没有注意力权重数据，无法可视化")
        return

    # 确保特征名称与实际数据维度匹配
    actual_feature_count = attention_weights[0].shape[1]
    if len(feature_names) != actual_feature_count:
        print(f"警告：特征名称数量 ({len(feature_names)}) 与实际数据维度 ({actual_feature_count}) 不匹配")
        # 调整特征名称列表至正确长度
        if len(feature_names) > actual_feature_count:
            feature_names = feature_names[:actual_feature_count]
        else:
            # 扩展特征名称列表
            for i in range(len(feature_names), actual_feature_count):
                feature_names.append(f'feature_{i+1}')

    # 计算平均注意力权重
    mean_attention = np.mean(attention_weights, axis=0)
    mean_attention = mean_attention.reshape(-1, len(feature_names))

    plt.figure(figsize=(12, 6))
    sns.heatmap(mean_attention, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=feature_names,
                yticklabels=['sample' + str(i + 1) for i in range(min(100, mean_attention.shape[0]))])
    plt.title('Enter feature attention weights', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Feature attention weights.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_predictions_with_confidence(test_true, test_pred, scaler, time_range, output_dir):
    """可视化预测结果和置信区间"""
    # 反归一化
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    true = scaler.inverse_transform(test_true.reshape(-1, 1))
    pred = scaler.inverse_transform(test_pred.reshape(-1, 1))

    # 计算误差
    errors = np.abs(true - pred)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # 创建置信区间
    upper_bound = pred + 1.96 * std_error
    lower_bound = pred - 1.96 * std_error

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    # 绘制预测结果
    plt.figure(figsize=(14, 7))

    # 添加置信区间
    plt.fill_between(range(len(pred)), lower_bound.flatten(), upper_bound.flatten(),
                     color='skyblue', alpha=0.4, label='95% confidence interval')

    # 绘制真实值和预测值
    plt.plot(true, 'o-', color='#2c7bb6', label='True value', alpha=0.8, markersize=4)
    plt.plot(pred, 'o-', color='#d7191c', label='Predicted value', alpha=0.8, markersize=4)

    plt.title(f'Groundwater level prediction results\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}', fontsize=14)
    plt.xlabel('Time step')
    plt.ylabel('Groundwater table')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(pred))

    # 添加误差信息文本框
    textstr = f'Mean Error: {mean_error:.2f}\nSTD Error: {std_error:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=props, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Prediction results and confidence intervals.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制误差分布
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=20, color='skyblue')
    plt.axvline(mean_error, color='red', linestyle='--', label=f'Average error: {mean_error:.2f}')
    plt.title('Prediction error distribution', fontsize=14)
    plt.xlabel('absolute error')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Prediction error distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制散点图比较真实值和预测值
    plt.figure(figsize=(8, 8))
    plt.scatter(true, pred, alpha=0.6, c='#4daf4a')

    # 添加理想线（y=x）
    min_val = min(np.min(true), np.min(pred))
    max_val = max(np.max(true), np.max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title('True vs. Predicted', fontsize=14)
    plt.xlabel('Real water table')
    plt.ylabel('Predict the water table.')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Scatter plot of true vs predicted values.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_wiring_diagram(model, output_dir):
    """可视化模型连接结构"""
    plt.figure(figsize=(12, 10))
    legend_handles = model.wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    plt.title('LTC model neuronal connectivity structure', fontsize=14)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LTC connection structure.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建NetworkX图可视化
    G = nx.DiGraph()

    # 添加神经元节点
    for i in range(model.wiring.units):
        G.add_node(f'N{i}', type='neuron')

    # 添加输入节点 - 修改这里，根据是否使用水平衡方程添加额外的特征名
    feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']
    if model.use_wb_equations:
        feature_names.append('wb_output')  # 如果使用水平衡方程，添加水平衡输出作为额外特征

    for i, name in enumerate(feature_names):
        G.add_node(name, type='input')

    # 添加输出节点
    G.add_node('output', type='output')

    # 添加连接边
    # 从输入到神经元
    sensory_adj = model.wiring.sensory_adjacency_matrix
    for i in range(min(len(feature_names), sensory_adj.shape[0])):  # 使用较小的范围，避免索引越界
        for j in range(sensory_adj.shape[1]):  # 神经元
            if sensory_adj[i, j] != 0:
                G.add_edge(feature_names[i], f'N{j}', weight=abs(sensory_adj[i, j]))

    # 神经元之间
    adj = model.wiring.adjacency_matrix
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                G.add_edge(f'N{i}', f'N{j}', weight=abs(adj[i, j]))

    # 从神经元到输出
    for i in range(model.wiring.output_dim):
        G.add_edge(f'N{i}', 'output', weight=1)

    # 创建基于类型的节点颜色映射
    color_map = {'input': 'skyblue', 'neuron': 'lightgreen', 'output': 'salmon'}
    node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]

    # 创建可视化
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # 根据权重计算边的宽度
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           arrowstyle='->', arrowsize=15)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    plt.title("LTC neuron connectivity network diagram", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LTC network diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_water_balance_equations(model, output_dir):
    """可视化水平衡方程组件及参数"""
    if not hasattr(model.ltc.rnn_cell, 'wb_params'):
        print("模型没有水平衡方程参数，无法可视化")
        return

    # 提取水平衡参数
    wb_params = {name: param.item() for name, param in model.ltc.rnn_cell.wb_params.items()}

    # 创建水平衡方程示意图
    plt.figure(figsize=(12, 10))

    # 绘制水平衡流程图
    ax = plt.subplot(111)

    # 定义组件位置和大小 - 新增岩溶含水层
    components = {
        'precipitation': (0.5, 0.95, 0.2, 0.1),  # x, y, width, height
        'interception': (0.5, 0.80, 0.2, 0.1),
        'surface': (0.5, 0.65, 0.2, 0.1),
        'preferential': (0.3, 0.50, 0.2, 0.1),
        'capillary': (0.7, 0.50, 0.2, 0.1),
        'gravity': (0.5, 0.35, 0.3, 0.1),  # 第四纪覆盖层重力水库
        'karst': (0.5, 0.20, 0.3, 0.1),   # 新增：岩溶含水层
        'output': (0.5, 0.05, 0.2, 0.1)
    }

    # 绘制组件框
    for name, (x, y, w, h) in components.items():
        color = plt.cm.viridis(0.1 if name == 'output' else
                               0.4 if name == 'karst' else
                               0.7)
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=True,
                             color=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, name.capitalize(), ha='center', va='center', fontsize=12)

    # 绘制箭头连接 - 添加岩溶含水层连接
    arrows = [
        ('precipitation', 'interception', "precipitation"),
        ('interception', 'surface', "Direct to surface precipitation"),
        ('surface', 'preferential', f"Priority streams ({wb_params['Rpr']:.2f})"),
        ('surface', 'capillary', f"Capillary permeation ({wb_params['Rin']:.2f})"),
        ('preferential', 'gravity', f"Pref. flow ({wb_params['Qpmax']:.2f})"),
        ('capillary', 'gravity', f"Cap. flow - ET ({wb_params['kl']:.2f})"),
        ('gravity', 'karst', f"Infiltration to karst ({wb_params['karst_infiltration']:.2f})"),
        ('karst', 'output', f"Karst to groundwater ({wb_params['karst_connect']:.2f})")
    ]

    for start, end, label in arrows:
        start_x, start_y = components[start][0], components[start][1] - components[start][3] / 2
        end_x, end_y = components[end][0], components[end][1] + components[end][3] / 2

        # 如果起点和终点x坐标不同，绘制曲线箭头
        if start_x != end_x:
            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
        else:
            ax.arrow(start_x, start_y, 0, end_y - start_y - 0.02,
                     head_width=0.02, head_length=0.02, fc='black', ec='black', lw=1.5)

        # 计算箭头中点位置放置标签
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # 添加偏移以避免与箭头重叠
        if start_x != end_x:
            offset_x = 0.05 * (-1 if end_x < start_x else 1)
            ax.text(mid_x + offset_x, mid_y, label, ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        else:
            ax.text(mid_x + 0.08, mid_y, label, ha='left', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # 添加关键参数文本框 - 更新以包含新的岩溶参数
    param_text = "\n".join([
        f"karst_connect: {wb_params['karst_connect']:.2f}",
        f"karst_storage: {wb_params['karst_storage']:.2f}",
        f"karst_infiltration: {wb_params['karst_infiltration']:.2f}",
        f"interface_release: {wb_params['interface_release']:.2f}",
        f"Rin: {wb_params['Rin']:.2f}",
        f"Rpr: {wb_params['Rpr']:.2f}",
        f"kl: {wb_params['kl']:.2f}",
        f"kn: {wb_params['kn']:.2f}",
        f"pumping_factor: {wb_params['pumping_factor']:.2f}"
    ])

    plt.text(0.85, 0.8, "Key parameters", fontsize=12, fontweight='bold',
             bbox=dict(facecolor='wheat', alpha=0.5))
    plt.text(0.85, 0.65, param_text, fontsize=10, va='top',
             bbox=dict(facecolor='wheat', alpha=0.5))

    # 添加地质分层示意
    plt.axhline(y=0.275, color='brown', linestyle='--', alpha=0.7)
    plt.text(0.95, 0.275, "Quaternary-Karst interface",
             fontsize=10, va='center', ha='right',
             bbox=dict(facecolor='white', alpha=0.7))

    plt.title('Water Balance Equation Components and Flow Diagrams', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Flow diagram of the water balance equation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_visualization_dashboard(model, dataloader, test_pred, test_true, scaler, output_dir):
    """创建完整的可视化仪表盘"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("开始创建可视化仪表盘...")

    # 1. 可视化模型结构
    print("1/8 可视化模型连接结构...")
    visualize_wiring_diagram(model, output_dir)

    # 2. 可视化预测结果
    print("2/8 可视化预测结果和置信区间...")
    # 假设时间范围与测试集大小一致
    time_range = range(len(test_true))
    visualize_predictions_with_confidence(test_true, test_pred, scaler, time_range, output_dir)

    # 3. 计算特征重要性
    print("3/8 计算并可视化特征重要性...")
    feature_importance = model.compute_feature_importance(dataloader)
    visualize_feature_importance(model, output_dir)

    # 4. 捕获神经元激活和水平衡状态
    print("4/8 捕获神经元状态和水平衡状态...")
    # 从dataloader中获取一小批数据用于可视化
    for x, y, times in dataloader:
        # 限制样本数量以加快处理速度
        if x.size(0) > 10:
            x = x[:10]
            times = times[:10]

        activations, wb_states, attention_weights = model.capture_neuron_activations(x, times)

        # 5. 可视化神经元动态
        print("5/8 可视化神经元动态...")
        visualize_neuron_dynamics(activations, model.wiring, output_dir)

        # 6. 可视化水平衡状态
        print("6/8 检查是否可视化水平衡状态...")
        if model.use_wb_equations and wb_states is not None:
            print("可视化水平衡状态变化...")
            if len(wb_states) > 0:  # 确保有数据
                visualize_water_balance(wb_states, times.cpu().numpy(), x.cpu().numpy(), output_dir)
            else:
                print("跳过水平衡状态可视化，因为wb_states为空")
        else:
            print("跳过水平衡状态可视化，因为水平衡方程未启用或wb_states为None")

        # 7. 可视化注意力权重
        # 7. 可视化注意力权重
        print("7/8 可视化注意力权重...")
        if model.use_wb_equations:
            feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp', 'wb']
        else:
            feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']  # 不包含水平衡输出
        visualize_attention_weights(attention_weights, feature_names, output_dir)

        # 只处理一批数据
        break

    # 8. 可视化水平衡方程
    # 8. 可视化水平衡方程
    print("8/8 检查是否可视化水平衡方程组件...")
    if hasattr(model, 'use_wb_equations') and model.use_wb_equations:
        print("可视化水平衡方程组件...")
        visualize_water_balance_equations(model, output_dir)
    else:
        print("跳过水平衡方程可视化，因为水平衡方程未启用")

    print(f"可视化仪表盘创建完成，结果保存在: {output_dir}")

    # 创建结果索引HTML
    create_html_index(output_dir)


def create_html_index(output_dir):
    """创建HTML结果索引页面"""
    # 获取所有PNG文件
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

    # 创建HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LTC模型与水平衡方程可视化结果</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333366; text-align: center; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; }
            .vis-item { margin: 15px; background-color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .vis-item img { max-width: 100%; height: auto; }
            .vis-item h3 { padding: 10px; margin: 0; background-color: #4472C4; color: white; }
            .vis-item p { padding: 10px; margin: 0; }
            .description { max-width: 800px; margin: 20px auto; padding: 15px; background-color: white; 
                          border-left: 5px solid #4472C4; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <h1>LTC模型与水平衡方程可视化结果</h1>

        <div class="description">
            <p>本可视化仪表盘展示了结合水平衡方程的LTC (Liquid Time-Constant) 神经网络模型在地下水位预测任务中的内部工作机制。
            通过这些可视化，我们可以更好地理解模型如何利用气象数据和水文过程进行预测，并洞悉影响预测结果的关键因素。</p>
        </div>

        <div class="container">
    """

    # 为每个图像添加项目
    for png_file in sorted(png_files):
        # 根据文件名生成标题和描述
        title = png_file.replace('.png', '').replace('_', ' ')

        # 根据文件名生成描述
        description = ""
        if "Feature importance analysis" in png_file:
            description = "Shows the extent to which each input feature influences the model's prediction results to help identify key drivers."
        elif "Neuronal dynamics" in png_file:
            description = "Shows the activation pattern of neurons over time, reflecting the change of the internal state of the model with the input sequence."
        elif "Water balance state" in png_file:
            description = "The changes of the states of each component of the water balance equation with time are displayed, and the dynamics of hydrological processes are revealed."
        elif "Predict the outcome" in png_file:
            description = "Shows how the model's predictions compare to the true value, including confidence intervals to reflect the prediction uncertainty."
        elif "LTC connection structure" in png_file:
            description = "The connection structure between neurons in the LTC model is demonstrated, reflecting the information flow path."
        elif "Water balance equation" in png_file:
            description = "Demonstrate the components and parameters of the water balance equation and explain the core processes of hydrological simulation."
        elif "Characteristic attention" in png_file:
            description = "Demonstrate the degree to which the model pays attention to different input features and reveal the decision-making process."
        else:
            description = "The visualisation shows important aspects of the model's internal structure and how it works."

        # 添加到HTML
        html_content += f"""
        <div class="vis-item">
            <h3>{title}</h3>
            <img src="{png_file}" alt="{title}">
            <p>{description}</p>
        </div>
        """

    # 完成HTML
    html_content += """
        </div>
    </body>
    </html>
    """

    # 写入文件
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML索引页面已创建: {os.path.join(output_dir, 'index.html')}")

def plot_predictions(true, pred, title, scaler):
    # 反归一化
    true = scaler.inverse_transform(true.reshape(-1, 1))
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    # 计算指标
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    plt.figure(figsize=(12, 6))
    plt.plot(true, label="True Values", color="#2c7bb6", alpha=0.8)
    plt.plot(pred, label="Predictions", color="#d7191c", linestyle="--", alpha=0.8)
    plt.title(f"{title}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


###########################
# 主程序
###########################

def main():
    # 数据准备
    dm = TrafficDataModule(seq_len=36, batch_size=32)
    dm.prepare_data()

    # 设置是否使用水平衡方程
    use_wb_equations = True

    # 使用增强版模型 - 替换原始的LTCForecaster
    model = LTCForecasterWithVisualization(input_size=6, ncp_units=8, lr=0.0005, use_wb_equations=use_wb_equations)

    # 可视化模型连接结构
    output_dir = "D:\\研一下\\LNNS预测\\LTC_多变量试验\\可视化结果_wb"
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("white")
    plt.figure(figsize=(6, 4))
    legend_handles = model.wiring.draw_graph(
        draw_labels=True,
        neuron_colors={"command": "tab:cyan"}
    )
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Model structure.png"), dpi=300)
    plt.close()

    # 训练配置
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=True,
        mode="min"
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )

    # 修改数据加载器，增加对时间间隔的处理
    dm.train_dataloader = lambda: DataLoader(
        TensorDataset(
            torch.tensor(dm.train_x, dtype=torch.float32),
            torch.tensor(dm.train_y, dtype=torch.float32),
            torch.tensor(dm.train_times, dtype=torch.float32)
        ),
        batch_size=dm.batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last =False # 确保不丢弃批次
    )

    dm.val_dataloader = lambda: DataLoader(
        TensorDataset(
            torch.tensor(dm.valid_x, dtype=torch.float32),
            torch.tensor(dm.valid_y, dtype=torch.float32),
            torch.tensor(dm.valid_times, dtype=torch.float32)
        ),
        batch_size=dm.batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=False
    )#yanrong

    dm.test_dataloader = lambda: DataLoader(
        TensorDataset(
            torch.tensor(dm.test_x, dtype=torch.float32),
            torch.tensor(dm.test_y, dtype=torch.float32),
            torch.tensor(dm.test_times, dtype=torch.float32)
        ),
        batch_size=dm.batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last=False
    )

    trainer = pl.Trainer(
        max_epochs=400,
        callbacks=[early_stop, checkpoint],
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger("logs/"),
        log_every_n_steps=5
    )

    # 训练模型
    print("开始训练模型...")
    trainer.fit(model, dm)
    print("模型训练完成！")

    # 加载最佳模型 - 使用增强版模型类
    best_model = LTCForecasterWithVisualization.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        input_size=6,
        ncp_units=8,
        lr=0.0005,
        use_wb_equations=use_wb_equations
    )
    best_model.eval()

    # 预测函数
    def get_predictions(loader):
        preds, trues = [], []
        with torch.no_grad():
            for x, y, times in loader:
                y_hat = best_model(x, times)
                preds.append(y_hat)
                trues.append(y)
        return torch.cat(preds).numpy(), torch.cat(trues).numpy()

    # 训练集预测
    print("生成训练集预测...")
    train_pred, train_true = get_predictions(dm.train_dataloader())
    # 使用增强版可视化
    visualize_predictions_with_confidence(
        train_true.reshape(-1, 1),
        train_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(train_true)),
        os.path.join(output_dir, "训练集")
    )

    # 保存预测结果
    train_true_inv = dm.scaler_y.inverse_transform(train_true.reshape(-1, 1))
    train_pred_inv = dm.scaler_y.inverse_transform(train_pred.reshape(-1, 1))
    pd.DataFrame({"True_value": train_true_inv.flatten(), "Predicted_value": train_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "训练集预测结果.xlsx"), index=False)

    # 验证集预测
    print("生成验证集预测...")
    val_pred, val_true = get_predictions(dm.val_dataloader())
    visualize_predictions_with_confidence(
        val_true.reshape(-1, 1),
        val_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(val_true)),
        os.path.join(output_dir, "验证集")
    )

    val_true_inv = dm.scaler_y.inverse_transform(val_true.reshape(-1, 1))
    val_pred_inv = dm.scaler_y.inverse_transform(val_pred.reshape(-1, 1))
    pd.DataFrame({"Val_value": val_true_inv.flatten(), "Predicted_value": val_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "验证集预测结果.xlsx"), index=False)

    # 测试集预测
    print("生成测试集预测...")
    test_pred, test_true = get_predictions(dm.test_dataloader())
    visualize_predictions_with_confidence(
        test_true.reshape(-1, 1),
        test_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(test_true)),
        os.path.join(output_dir, "测试集")
    )

    test_true_inv = dm.scaler_y.inverse_transform(test_true.reshape(-1, 1))
    test_pred_inv = dm.scaler_y.inverse_transform(test_pred.reshape(-1, 1))
    pd.DataFrame({"Test_value": test_true_inv.flatten(), "Predicted_value": test_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "测试集预测结果.xlsx"), index=False)

    # 创建完整的可视化仪表盘
    print("创建综合可视化仪表盘...")
    vis_output_dir = os.path.join(output_dir, "可视化仪表盘")
    create_visualization_dashboard(
        best_model,
        dm.test_dataloader(),
        test_pred.reshape(-1, 1),
        test_true.reshape(-1, 1),
        dm.scaler_y,
        vis_output_dir
    )

    print(f"所有分析和可视化已完成，结果保存在: {output_dir}")
    print(f"可视化仪表盘位置: {vis_output_dir}")
    print(f"请打开 {os.path.join(vis_output_dir, 'index.html')} 查看完整的可视化报告")


if __name__ == "__main__":
    main()