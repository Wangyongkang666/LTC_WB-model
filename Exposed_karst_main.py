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
# Setting global styles
sns.set(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "DejaVu Sans"


###########################
# Data loading and preprocessing
###########################

def load_trace():
    # Assume that the data file is located in the data/traffic directory of the current directory
    df = pd.read_csv("**********")

    time_intervals = df["time_interval"].values.astype(np.float32) / 5.0
    # Processing Features
    pre = df["pre"].values.astype(np.float32)
    pre -= np.mean(pre)
    pre /= np.std(pre)

    # Adding temperature data
    temperature = df["temperature"].values.astype(np.float32)
    temperature -= np.mean(temperature)
    temperature /= np.std(temperature)

    # Time feature processing
    date_time = [datetime.strptime(d, "%Y/%m/%d %H:%M") for d in df["date_time"]]
    day_of_year = [d.timetuple().tm_yday for d in date_time]
    day_length = []
    for doy in day_of_year:
        # Shift the dates so that the winter solstice (approximately day 355) corresponds to the sine wave minimum
        shifted_day = (doy - 83) % 365 
        angle = (2 * math.pi * shifted_day) / 365
        relative_length = (math.sin(angle) + 1) / 2
        day_length.append(relative_length)

    # Convert to numpy array with float32 type
    day_length = np.array(day_length).astype(np.float32)

    # Add groundwater withdrawal data, if not available, build simulated values ​​based on seasonality or other factors
    if "ext" in df.columns:
        extraction = df["ext"].values.astype(np.float32)
        extraction -= np.mean(extraction)
        extraction /= np.std(extraction)
    else:
        # Simulation value: large mining volume in summer and small mining volume in winter
        extraction = np.array([0.7 if 5 <= d.month <= 9 else 0.3 for d in date_time]).astype(np.float32)
        extraction -= np.mean(extraction)
        extraction /= np.std(extraction)
        # Add two new features
    if "srv" in df.columns:
        feature5 = df["srv"].values.astype(np.float32)
        feature5 -= np.mean(feature5)
        feature5 /= np.std(feature5)
    else:
        feature5 = np.zeros_like(pre)

    if "sp" in df.columns:
        feature6 = df["sp"].values.astype(np.float32)
        feature6 -= np.mean(feature6)
        feature6 /= np.std(feature6)
    else:
        feature6 = np.zeros_like(pre)

        # Combination Features
    features = np.stack([pre, temperature, day_length, extraction, feature5, feature6], axis=-1)

    # Processing target variables
    well_level = df["well_level"].values.astype(np.float32).reshape(-1, 1)

    return features, well_level, time_intervals


def create_sequences(features, target, time_intervals, seq_len=12, inc=1):
    sequences_x = []
    sequences_y = []
    sequences_time = []  # The time interval used to store the last step of each sequence

    for s in range(0, len(features) - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(features[start:end])
        sequences_y.append(target[end - 1])
        sequences_time.append(time_intervals[start:end])  # Add the corresponding time interval sequence

    return (np.stack(sequences_x, axis=0),
            np.array(sequences_y).reshape(-1, 1),
            np.stack(sequences_time, axis=0))  # Returns a sequence of time intervals

###########################
# Data Module
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
        self.train_times = np.stack(train_times, axis=0)  # Storage time interval
        total_seqs = self.train_x.shape[0]
        print("Total number of training sequences: {}".format(total_seqs))
        valid_size = int(0.15 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.test_x = self.train_x[-test_size:]
        self.test_y = self.train_y[-test_size:]
        self.test_times = self.train_times[-test_size:]  # Add test set interval
        self.valid_x = self.train_x[-(test_size + valid_size):-test_size]
        self.valid_y = self.train_y[-(test_size + valid_size):-test_size]
        self.valid_times = self.train_times[-(test_size + valid_size):-test_size]  # Add validation set interval
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
# LTC Model
###########################

class LTCForecaster(pl.LightningModule):
    def __init__(self, input_size=6, ncp_units=32, lr=0.01, use_wb_equations=True, mixed_memory=True):
        """
        Initialize the LTC prediction model

        parameter:
        input_size: The number of original input features
        ncp_units: Number of neural network units
        lr: Learning Rate
        use_wb_equations: Whether to use the water balance equation
        """
        super().__init__()
        self.lr = lr
        self.input_size = input_size
        self.use_wb_equations = use_wb_equations
        self.mixed_memory = mixed_memory  # Add mixed_memory attribute

        # Creating a neuron wiring configuration
        self.wiring = AutoNCP(ncp_units, 1, sparsity_level=0.5)

        # 为wiring.build()Calculate the correct input dimensions
        extended_input_size = input_size + 1 if use_wb_equations else input_size

        # Use the expanded input dimensions to construct wiring
        self.wiring.build(extended_input_size)

        if self.mixed_memory:
            self.lstm = nn.LSTMCell(
                input_size=extended_input_size,
                hidden_size=self.wiring.units
            )
        # Prepare input dimensions for LTC
        self.ltc = LTC(
            input_size=extended_input_size,
            units=self.wiring,
            return_sequences=False,
            batch_first=True,
            mixed_memory=self.mixed_memory,
            use_wb_equations=use_wb_equations
        )
        # Used to track training and validation losses
        self.train_loss = []
        self.val_loss = []

    def forward(self, x, time_intervals=None):
        # Get the batch size
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Calculating neural state size
        neural_state_size = self.wiring.units

        # Initialize different states depending on whether the water balance equation is used
        if self.use_wb_equations:
            # The water balance state requires 4 additional state variables
            total_state_size = neural_state_size + 1
            h_state = torch.zeros(batch_size, total_state_size, device=x.device)
        else:
            h_state = torch.zeros(batch_size, neural_state_size, device=x.device)

        # Create LSTM cell state c0 (only for neuron state size)
        c_state = torch.zeros(batch_size, neural_state_size, device=x.device)

        # Combine them into a tuple
        initial_state = (h_state, c_state)

        # Use the output that stores all time steps
        outputs = []

        # The current state is initialized to the initial state
        current_state = initial_state

        # Processing a sequence time-step by time
        for t in range(seq_len):
            # Get the input for the current time step
            x_t = x[:, t, :]
            if len(x_t.shape) == 1:
                x_t = x_t.unsqueeze(0)
            # Get the time interval of the current time step
            if time_intervals is not None:
                elapsed_time = time_intervals[:, t]
            else:
                elapsed_time = torch.ones(batch_size, device=x.device)  # 默认为1.0

            # Dealing with the water balance equation
            if self.use_wb_equations:
                h_state, c_state = current_state

                # Separating neuronal state and water balance from hidden state
                neuron_state = h_state[:, :neural_state_size]
                wb_state = h_state[:, neural_state_size:]

                # Only the first 4 features are used to calculate water balance
                wb_inputs = x_t[:, :4]
                # Calculate water balance output and obtain new water balance status
                wb_output, new_wb_state = self.ltc.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)
                # Append the water balance output to the input, creating an enhanced input
                augmented_inputs = torch.cat([x_t, wb_output], dim=1)

                # For LSTM in hybrid memory mode, use the original neuron state
                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                # Forward propagation of LTC units, using the current time interval
                output, new_h_state = self.ltc.rnn_cell.forward(
                    augmented_inputs,
                    torch.cat([neuron_state, wb_state], dim=1),
                    elapsed_time=elapsed_time
                )

                # Update the water balance state part in h_state (adjust according to the actual return structure)
                if self.use_wb_equations:
                    # Assume that new_h_state is already the integrated state
                    new_h_state = torch.cat([
                        new_h_state[:, :neural_state_size],
                        new_wb_state
                    ], dim=1)

                # Update current status
                current_state = (new_h_state, c_state)
            else:
                # Raw LTC processing, using the current time interval
                h_state, c_state = current_state

                # For LSTM hybrid memory mode
                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(x_t, (h_state, c_state))
                    h_state = neuron_state

                # Use LTC's rnn_cell instead of self.rnn_cell
                output, new_h_state = self.ltc.rnn_cell.forward(x_t, h_state, elapsed_time=elapsed_time)

                # Update current status
                current_state = (new_h_state, c_state)

            outputs.append(output)

        # Returns the output of the last time step or the entire sequence
        if self.ltc.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return outputs[-1]

    def training_step(self, batch, batch_idx):
        x, y, time_intervals = batch  # Updated to handle time intervals
        y_hat = self(x, time_intervals)  # Delivery time interval
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_loss.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, time_intervals = batch  # Updated to handle time intervals
        y_hat = self(x, time_intervals)  # Delivery time interval
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


# Add to the model class of the original code
class LTCForecasterWithVisualization(LTCForecaster):
    """
    Expand the LTC prediction model and add visualization capabilities
    """

    def __init__(self, input_size=6, ncp_units=32, lr=0.01, use_wb_equations=True, mixed_memory=True):
        super().__init__(input_size, ncp_units, lr, use_wb_equations, mixed_memory)
        # Record additional information during training
        self.feature_importance = {}
        self.neuron_activations = []
        self.water_balance_states = []
        self.attention_weights = []

    def capture_neuron_activations(self, x, time_intervals=None):
        """Capturing neuronal activation and water balance"""
        # Get the batch size
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialization state
        if self.use_wb_equations:
            total_state_size = self.wiring.units + 1
            h_state = torch.zeros(batch_size, total_state_size, device=x.device)
        else:
            h_state = torch.zeros(batch_size, self.wiring.units, device=x.device)

        c_state = torch.zeros(batch_size, self.wiring.units, device=x.device)

        initial_state = (h_state, c_state)
        current_state = initial_state

        # Record neuronal activation and water balance status at each time step
        activations = []
        wb_states = None  # If water balance is not used, initialize to None
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

                # Calculating water balance
                wb_inputs = x_t[:, :4]
                wb_output, new_wb_state = self.ltc.rnn_cell._wb_model_solve(wb_inputs, wb_state, batch_size=batch_size)

                # Maintain water balance
                wb_states.append(new_wb_state.detach().cpu().numpy())

                # Creating Enhanced Input
                augmented_inputs = torch.cat([x_t, wb_output], dim=1)

                if self.mixed_memory:
                    neuron_state, c_state = self.lstm(augmented_inputs, (neuron_state, c_state))

                # Calculate attention weights (approximately the correlation between input and neuron state)
                attention = torch.zeros((batch_size, augmented_inputs.size(1)), device=x.device)

                for i in range(augmented_inputs.size(1)):
                    corr = torch.mean(augmented_inputs[:, i].unsqueeze(1) * neuron_state, dim=1)
                    attention[:, i] = corr
                attention = torch.softmax(attention, dim=1)
                attn_weights.append(attention.detach().cpu().numpy())

                # Forward Propagation
                output, new_h_state = self.ltc.rnn_cell.forward(
                    augmented_inputs,
                    torch.cat([neuron_state, wb_state], dim=1),
                    elapsed_time=elapsed_time
                )

                # Save neuron activations
                activations.append(new_h_state[:, :self.wiring.units].detach().cpu().numpy())

                # Update Status
                new_h_state = torch.cat([
                    new_h_state[:, :self.wiring.units],
                    new_wb_state
                ], dim=1)

                current_state = (new_h_state, c_state)
            else:
                # When the water balance equation is not used
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
        """Calculating feature importance"""
        feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']
        importances = {name: [] for name in feature_names}

        with torch.no_grad():
            for x, y, times in dataloader:
                # Calculate initial forecast
                original_pred = self(x, times)

                # Perturb each feature and calculate the impact
                for i, feature_name in enumerate(feature_names):
                    # Creating perturbed data
                    perturbed_x = x.clone()
                    perturbed_x[:, :, i] = torch.mean(perturbed_x[:, :, i])

                    # Predicting using perturbed data
                    perturbed_pred = self(perturbed_x, times)

                    # Calculate impact (using MSE change)
                    impact = torch.mean(torch.abs(original_pred - perturbed_pred)).item()
                    importances[feature_name].append(impact)

        # Calculating average impact
        for feature_name in feature_names:
            self.feature_importance[feature_name] = np.mean(importances[feature_name])

        # Normalization
        total = sum(self.feature_importance.values())
        for feature_name in feature_names:
            self.feature_importance[feature_name] /= total

        return self.feature_importance

###########################
# Visualization function
###########################
# Visualization function
def visualize_feature_importance(model, output_dir):
    """Visualizing feature importance"""
    if not hasattr(model, 'feature_importance') or not model.feature_importance:
        print("The model does not calculate feature importance and cannot be visualized")
        return

    plt.figure(figsize=(10, 6))
    features = list(model.feature_importance.keys())
    importances = list(model.feature_importance.values())

    # Create a horizontal bar chart
    bars = plt.barh(features, importances, color=plt.cm.viridis(np.linspace(0, 0.8, len(features))))

    # Add value labels
    for i, v in enumerate(importances):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.title('Feature importance analysis', fontsize=14)
    plt.xlabel('Importance Score (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Feature importance analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_neuron_dynamics(activations, wiring, output_dir, step=1):
    """Visualizing neuronal dynamics"""
    if activations is None or len(activations) == 0:
        print("No neuron activation data, no visualization")
        return

    # Select a subset of time steps for visualization
    time_steps = list(range(0, len(activations), step))
    if time_steps[-1] != len(activations) - 1:
        time_steps.append(len(activations) - 1)

    plt.figure(figsize=(15, 10))

    # Calculate the average activation
    mean_activations = np.mean(activations, axis=1)

    # Create a time series line for each neuron
    for i in range(min(wiring.units, 10)):  # Limit to the first 10 neurons to avoid overcrowding
        plt.plot(mean_activations[:, i], label=f'neuron {i + 1}', alpha=0.7)

    # Marking key time steps
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
    """Visualize the state changes of the water balance equation"""
    if wb_states is None or len(wb_states) == 0:
        print("No water balance status data, no visualization")
        return

    # Water balance component name
    wb_components = ['karset Reservoir']

    # Calculate the average state
    mean_wb_states = np.mean(wb_states, axis=1)

    # Plotting changes in water balance components
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

    # Plotting water balance components versus input characteristics
    # Process input data to obtain precipitation and temperature
    mean_inputs = np.mean(input_data, axis=0)
    precipitation = mean_inputs[:, 0]  # Assume precipitation is the first feature
    temperature = mean_inputs[:, 1]  # Assume temperature is the second feature

    # Draw multi-panel charts
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2])

    # Precipitation map
    ax1 = plt.subplot(gs[0])
    ax1.bar(range(len(precipitation)), precipitation, color='skyblue', alpha=0.7)
    ax1.set_title('Precipitation', fontsize=12)
    ax1.set_ylabel('Precipitation (Normalised)')
    ax1.grid(True, alpha=0.3)

    # Temperature map
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(temperature, color='red', linewidth=2)
    ax2.set_title('Temperature', fontsize=12)
    ax2.set_ylabel('Temperature (Normalised)')
    ax2.grid(True, alpha=0.3)

    # Water balance diagram
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
    """Visualizing Attention Weights"""
    if attention_weights is None or len(attention_weights) == 0:
        print("No attention weight data, no visualization")
        return

    # Make sure feature names match actual data dimensions
    actual_feature_count = attention_weights[0].shape[1]
    if len(feature_names) != actual_feature_count:
        print(f"Warning: Number of feature names ({len(feature_names)}) does not match actual data dimension ({actual_feature_count})")
        # 调Resize feature name list to correct length
        if len(feature_names) > actual_feature_count:
            feature_names = feature_names[:actual_feature_count]
        else:
            # Extended feature name list
            for i in range(len(feature_names), actual_feature_count):
                feature_names.append(f'feature_{i+1}')

    # Calculate the average attention weight
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
    """Visualize prediction results and confidence intervals"""
    # Denormalization
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    true = scaler.inverse_transform(test_true.reshape(-1, 1))
    pred = scaler.inverse_transform(test_pred.reshape(-1, 1))

    # Calculation error
    errors = np.abs(true - pred)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Creating confidence intervals
    upper_bound = pred + 1.96 * std_error
    lower_bound = pred - 1.96 * std_error

    # Calculating evaluation metrics
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)

    # Plotting the prediction results
    plt.figure(figsize=(14, 7))

    # Add confidence intervals
    plt.fill_between(range(len(pred)), lower_bound.flatten(), upper_bound.flatten(),
                     color='skyblue', alpha=0.4, label='95% confidence interval')

    # Plot the true and predicted values
    plt.plot(true, 'o-', color='#2c7bb6', label='True value', alpha=0.8, markersize=4)
    plt.plot(pred, 'o-', color='#d7191c', label='Predicted value', alpha=0.8, markersize=4)

    plt.title(f'Groundwater level prediction results\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}', fontsize=14)
    plt.xlabel('Time step')
    plt.ylabel('Groundwater table')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(pred))

    # Add error message text box
    textstr = f'Mean Error: {mean_error:.2f}\nSTD Error: {std_error:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.annotate(textstr, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=props, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Prediction results and confidence intervals.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting the error distribution
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

    # Draw a scatter plot to compare the true value and the predicted value
    plt.figure(figsize=(8, 8))
    plt.scatter(true, pred, alpha=0.6, c='#4daf4a')

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
    """Visualize model connection structure"""
    plt.figure(figsize=(12, 10))
    legend_handles = model.wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    plt.title('LTC model neuronal connectivity structure', fontsize=14)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LTC connection structure.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Creating a NetworkX graph visualization
    G = nx.DiGraph()

    # Adding Neuron Nodes
    for i in range(model.wiring.units):
        G.add_node(f'N{i}', type='neuron')

    # Add Input Node - Modify this to add additional feature names depending on whether the water balance equation is used
    feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']
    if model.use_wb_equations:
        feature_names.append('wb_output')  # If using the water balance equation, add the water balance output as an extra feature

    for i, name in enumerate(feature_names):
        G.add_node(name, type='input')

    # Adding an Output Node
    G.add_node('output', type='output')

    # Adding connecting edges
    # From input to neurons
    sensory_adj = model.wiring.sensory_adjacency_matrix
    for i in range(min(len(feature_names), sensory_adj.shape[0])):  # Use smaller ranges to avoid index out of bounds
        for j in range(sensory_adj.shape[1]):  # Neurons
            if sensory_adj[i, j] != 0:
                G.add_edge(feature_names[i], f'N{j}', weight=abs(sensory_adj[i, j]))

    # Between neurons
    adj = model.wiring.adjacency_matrix
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                G.add_edge(f'N{i}', f'N{j}', weight=abs(adj[i, j]))

    # From neurons to outputs
    for i in range(model.wiring.output_dim):
        G.add_edge(f'N{i}', 'output', weight=1)

    # Creating a type-based node color map
    color_map = {'input': 'skyblue', 'neuron': 'lightgreen', 'output': 'salmon'}
    node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]

    # Create visualizations
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    # Draw Node
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # Calculate the width of the edge based on the weight
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

    # Draw Edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           arrowstyle='->', arrowsize=15)

    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    plt.title("LTC neuron connectivity network diagram", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LTC network diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_water_balance_equations(model, output_dir):
    """Visualize components and parameters of exposed karst water balance equation"""
    if not hasattr(model.ltc.rnn_cell, 'wb_params'):
        print("The model has no water balance equation parameters and cannot be visualized")
        return

    # Extraction of water balance parameters
    wb_params = {name: param.item() for name, param in model.ltc.rnn_cell.wb_params.items()}

    # Create a water balance equation diagram
    plt.figure(figsize=(12, 10))

    # Draw a water balance flow chart
    ax = plt.subplot(111)

    # Define component position and size
    components = {
        'precipitation': (0.5, 0.8, 0.2, 0.1),  # x, y, width, height
        'surface': (0.5, 0.6, 0.2, 0.1),
        'karst': (0.5, 0.4, 0.3, 0.1),
        'output': (0.5, 0.2, 0.2, 0.1)
    }

    # Draw the component box
    for name, (x, y, w, h) in components.items():
        color = plt.cm.viridis(0.1 if name == 'output' else
                               0.4 if name == 'karst' else
                               0.7)
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, fill=True,
                             color=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, name.capitalize(), ha='center', va='center', fontsize=12)

    # Draw arrow connections
    arrows = [
        ('precipitation', 'surface', "Surface runoff"),
        ('surface', 'karst', f"Infiltration ({wb_params['karst_infiltration']:.2f})"),
        ('karst', 'output', f"Discharge ({wb_params['karst_release']:.2f})")
    ]

    for start, end, label in arrows:
        start_x, start_y = components[start][0], components[start][1] - components[start][3] / 2
        end_x, end_y = components[end][0], components[end][1] + components[end][3] / 2

        ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

        # Calculate the midpoint of the arrow and place the label
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        ax.text(mid_x + 0.05, mid_y, label, ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    # Add key parameter text box
    param_text = "\n".join([
        f"karst_storage: {wb_params['karst_storage']:.2f}",
        f"karst_infiltration: {wb_params['karst_infiltration']:.2f}",
        f"karst_release: {wb_params['karst_release']:.2f}",
        f"pumping_factor: {wb_params['pumping_factor']:.2f}"
    ])

    plt.text(0.8, 0.8, "Key parameters", fontsize=12, fontweight='bold',
             bbox=dict(facecolor='wheat', alpha=0.5))
    plt.text(0.8, 0.65, param_text, fontsize=10, va='top',
             bbox=dict(facecolor='wheat', alpha=0.5))

    plt.title('Water Balance Equation for Exposed Karst', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Exposed karst water balance diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_visualization_dashboard(model, dataloader, test_pred, test_true, scaler, output_dir):
    """Create complete visual dashboards"""
    # Create Output Directory
    os.makedirs(output_dir, exist_ok=True)

    print("Start creating a visualization dashboard...")

    # 1. Visualizing model structure
    print("1/8 Visualize model connection structure...")
    visualize_wiring_diagram(model, output_dir)

    # 2. Visualizing prediction results
    print("2/8 Visualize prediction results and confidence intervals...")
    # Assume the time range is consistent with the test set size
    time_range = range(len(test_true))
    visualize_predictions_with_confidence(test_true, test_pred, scaler, time_range, output_dir)

    # 3. Calculating feature importance
    print("3/8 Calculate and visualize feature importance...")
    feature_importance = model.compute_feature_importance(dataloader)
    visualize_feature_importance(model, output_dir)

    # 4. Capturing neuronal activation and water balance status
    print("4/8 Capturing neuronal states and water balance...")
    # Get a small batch of data from the dataloader for visualization
    for x, y, times in dataloader:
        # Limit the number of samples to speed up processing
        if x.size(0) > 10:
            x = x[:10]
            times = times[:10]

        activations, wb_states, attention_weights = model.capture_neuron_activations(x, times)

        # 5. Visualizing neuronal dynamics
        print("5/8 Visualizing neuronal dynamics...")
        visualize_neuron_dynamics(activations, model.wiring, output_dir)

        # 6. Visualize water balance status
        print("6/8 Check whether the water balance status is visualized...")
        if model.use_wb_equations and wb_states is not None:
            print("Visualize changes in water balance...")
            if len(wb_states) > 0: 
                visualize_water_balance(wb_states, times.cpu().numpy(), x.cpu().numpy(), output_dir)
            else:
                print("Skipping water balance state visualization because wb_states is empty")
        else:
            print("Skipping water balance state visualization because water balance equations are not enabled or wb_states is None")


        # 7. 可视化注意力权重
        print("7/8 Visualizing Attention Weights...")
        if model.use_wb_equations:
            feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp', 'wb']
        else:
            feature_names = ['pre', 'tep', 'day_length', 'ext', 'src', 'sp']
        visualize_attention_weights(attention_weights, feature_names, output_dir)

        break

    # 8. Visualizing the Water Balance Equation
    print("8/8 Check whether to visualize water balance equation components...")
    if hasattr(model, 'use_wb_equations') and model.use_wb_equations:
        print("Visualize the water balance equation component...")
        visualize_water_balance_equations(model, output_dir)
    else:
        print("Skipping water balance equation visualization because water balance equation is not enabled")

    print(f"The visualization dashboard is created and the results are saved in: {output_dir}")

    # Create the result index HTML
    create_html_index(output_dir)


def create_html_index(output_dir):
    # Get all PNG files
    png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

    # Creating HTML Content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualization results of LTC model and water balance equation</title>
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
        <h1>Visualization results of LTC model and water balance equation</h1>

        <div class="description">
            <p>This visualization dashboard shows the inner working mechanism of the LTC (Liquid Time-Constant) neural network model combined with the water balance equation in the task of groundwater level prediction.
            Through these visualizations, we can better understand how the model uses meteorological data and hydrological processes to make predictions and gain insight into the key factors that influence the forecast results.</p>
        </div>

        <div class="container">
    """

    # Add items for each image
    for png_file in sorted(png_files):
        # Generate title and description based on file name
        title = png_file.replace('.png', '').replace('_', ' ')

        # Generate description based on file name
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

        html_content += f"""
        <div class="vis-item">
            <h3>{title}</h3>
            <img src="{png_file}" alt="{title}">
            <p>{description}</p>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Writing to a file
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML index page created: {os.path.join(output_dir, 'index.html')}")

def plot_predictions(true, pred, title, scaler):
    # Denormalization
    true = scaler.inverse_transform(true.reshape(-1, 1))
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    # Calculation indicators
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
# Main Program
###########################

def main():
    # Data preparation
    dm = TrafficDataModule(seq_len=36, batch_size=32)
    dm.prepare_data()

    # Set whether to use the water balance equation
    use_wb_equations = True

    # Use the enhanced model - replace the original LTCForecaster
    model = LTCForecasterWithVisualization(input_size=6, ncp_units=8, lr=0.0005, use_wb_equations=use_wb_equations)

    # Visualize model connection structure
    output_dir = "******************"
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

    # Training Configuration
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

    # Modify the data loader to add time interval processing理
    dm.train_dataloader = lambda: DataLoader(
        TensorDataset(
            torch.tensor(dm.train_x, dtype=torch.float32),
            torch.tensor(dm.train_y, dtype=torch.float32),
            torch.tensor(dm.train_times, dtype=torch.float32)
        ),
        batch_size=dm.batch_size,
        num_workers=4,
        persistent_workers=True,
        drop_last = False
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
        drop_last= False
    )

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

    # Training the model
    print("Start training the model...")
    trainer.fit(model, dm)
    print("Model training completed！")

    # Loading the best model - Using the enhanced model class
    best_model = LTCForecasterWithVisualization.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        input_size=6,
        ncp_units=8,
        lr=0.0005,
        use_wb_equations=use_wb_equations
    )
    best_model.eval()

    # Prediction Function
    def get_predictions(loader):
        preds, trues = [], []
        with torch.no_grad():
            for x, y, times in loader:
                y_hat = best_model(x, times)
                preds.append(y_hat)
                trues.append(y)
        return torch.cat(preds).numpy(), torch.cat(trues).numpy()

    # Training set prediction
    print("Generating training set predictions...")
    train_pred, train_true = get_predictions(dm.train_dataloader())
    # Using enhanced visualization
    visualize_predictions_with_confidence(
        train_true.reshape(-1, 1),
        train_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(train_true)),
        os.path.join(output_dir, "Training set")
    )

    # Save prediction results
    train_true_inv = dm.scaler_y.inverse_transform(train_true.reshape(-1, 1))
    train_pred_inv = dm.scaler_y.inverse_transform(train_pred.reshape(-1, 1))
    pd.DataFrame({"True_value": train_true_inv.flatten(), "Predicted_value": train_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "Training set prediction results.xlsx"), index=False)

    # Validation set prediction
    print("Generate validation set predictions...")
    val_pred, val_true = get_predictions(dm.val_dataloader())
    visualize_predictions_with_confidence(
        val_true.reshape(-1, 1),
        val_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(val_true)),
        os.path.join(output_dir, "Validation set")
    )

    val_true_inv = dm.scaler_y.inverse_transform(val_true.reshape(-1, 1))
    val_pred_inv = dm.scaler_y.inverse_transform(val_pred.reshape(-1, 1))
    pd.DataFrame({"Val_value": val_true_inv.flatten(), "Predicted_value": val_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "Validation set prediction results.xlsx"), index=False)

    # Test set prediction
    print("Generate test set predictions...")
    test_pred, test_true = get_predictions(dm.test_dataloader())
    visualize_predictions_with_confidence(
        test_true.reshape(-1, 1),
        test_pred.reshape(-1, 1),
        dm.scaler_y,
        range(len(test_true)),
        os.path.join(output_dir, "Test Set")
    )

    test_true_inv = dm.scaler_y.inverse_transform(test_true.reshape(-1, 1))
    test_pred_inv = dm.scaler_y.inverse_transform(test_pred.reshape(-1, 1))
    pd.DataFrame({"Test_value": test_true_inv.flatten(), "Predicted_value": test_pred_inv.flatten()}).to_excel(
        os.path.join(output_dir, "Test set prediction results.xlsx"), index=False)

    # Create complete visual dashboards
    print("Create comprehensive visual dashboards...")
    vis_output_dir = os.path.join(output_dir, "Visualization dashboard")
    create_visualization_dashboard(
        best_model,
        dm.test_dataloader(),
        test_pred.reshape(-1, 1),
        test_true.reshape(-1, 1),
        dm.scaler_y,
        vis_output_dir
    )

    print(f"All analysis and visualization is done and the results are saved in: {output_dir}")
    print(f"Visualize dashboard location: {vis_output_dir}")
    print(f"请打开 {os.path.join(vis_output_dir, 'index.html')} View the full visual report")


if __name__ == "__main__":
    main()
