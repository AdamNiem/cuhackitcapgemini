import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os

# Directories
base_dir = '/Users/ethanmarquez/Downloads/Capgemini_Challenge/LSTm/'
training_dir = base_dir + '/Belt Training Data'
testing_dir = base_dir + '/Belt Test Data'
train_files = [f for f in os.listdir(training_dir) if f.endswith('.xlsx')]
test_files = [f for f in os.listdir(testing_dir) if f.endswith('.xlsx')]

# Expected ranges for data
expected_ranges = {
    'Vibration Frequency': (1490, 1510),
    'Vibration Amplitude': (0.04, 0.06),
    'Bearing Temperature': (60, 80),
    'Motor Temperature': (80, 100),
    'Belt Load': (1.0, 1.4),
    'Torque': (280, 320),
    'Noise Levels': (55, 65),
    'Current and Voltage': (14, 16),
    'Hydraulic Pressure': (375, 385),
    'Belt Thickness': (1.5, 1.7),
    'Roller Condition': (100, 65),
}

# Preprocess file
def preprocess_file(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"Columns in {file_path}: {df.columns.tolist()}")
        if 'Timestamp' not in df.columns:
            print(f"'Timestamp' not found in {file_path}. Skipping.")
            return None
        time_col = 'Timestamp'
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isna().all():
            print(f"No valid timestamps in {file_path}. Skipping.")
            return None
        df = df.dropna(subset=[time_col])

        # Mark out-of-range events
        first_out_of_range = None
        for col, (min_val, max_val) in expected_ranges.items():
            if col in df.columns:
                df[f'{col}_OutOfRange'] = ((df[col] < min_val) | (df[col] > max_val)).astype(int)
                if first_out_of_range is None and (df[f'{col}_OutOfRange'] == 1).any():
                    first_out_of_range = col

        # Target: time to 'Down' status, capped at 30 days
        df['Status'] = df['Status'].map({'Running': 0, 'Maintenance': 1, 'Down': 2})
        df['TimeToDown'] = float('inf')
        df['FirstOutOfRange'] = first_out_of_range
        down_idx = df[df['Status'] == 2].index
        for idx in df.index:
            if idx in down_idx:
                df.loc[idx, 'TimeToDown'] = 0
            else:
                future_down = down_idx[down_idx > idx]
                if not future_down.empty:
                    next_down = df.loc[future_down[0], time_col]
                    lag = (next_down - df.loc[idx, time_col]).total_seconds() / 60
                    df.loc[idx, 'TimeToDown'] = min(lag, 43200)  # Cap at 30 days

        # Features: include _OutOfRange and one-hot encoded FirstOutOfRange
        features = [f'{col}_OutOfRange' for col in expected_ranges if f'{col}_OutOfRange' in df.columns]
        df = pd.get_dummies(df, columns=['FirstOutOfRange'], drop_first=True)
        features.extend([col for col in df.columns if col.startswith('FirstOutOfRange_')])
        return df[features + ['TimeToDown', time_col]].dropna()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Combine data
train_dfs = [preprocess_file(os.path.join(training_dir, f)) for f in train_files]
train_dfs = [df for df in train_dfs if df is not None]
train_df = pd.concat(train_dfs, ignore_index=True)
train_df = train_df[train_df['TimeToDown'] != float('inf')]
print(f"Training data rows after filtering: {len(train_df)}")
print(f"Training TimeToDown stats (minutes): min={train_df['TimeToDown'].min():.2f}, max={train_df['TimeToDown'].max():.2f}, mean={train_df['TimeToDown'].mean():.2f}")

test_dfs = [preprocess_file(os.path.join(testing_dir, f)) for f in test_files]
test_dfs = [df for df in test_dfs if df is not None]
if not test_dfs:
    print("No valid test files found. Exiting.")
    exit()
test_df = pd.concat(test_dfs, ignore_index=True)
test_df = test_df[test_df['TimeToDown'] != float('inf')]
print(f"Testing data rows after filtering: {len(test_df)}")
print(f"Testing TimeToDown stats (minutes): min={test_df['TimeToDown'].min():.2f}, max={test_df['TimeToDown'].max():.2f}, mean={test_df['TimeToDown'].mean():.2f}")

# Check for 'Down' events
if len(train_df) == 0 or len(test_df) == 0:
    print("No 'Down' events found in training or testing data. Exiting.")
    exit()

# Transform TimeToDown with log
train_df['TimeToDown'] = np.log1p(train_df['TimeToDown'])
test_df['TimeToDown'] = np.log1p(test_df['TimeToDown'])

# Prepare sequences
sequence_length = 20
scaler = MinMaxScaler()
X_train, y_train = [], []

for i in range(len(train_df) - sequence_length):
    seq = train_df.drop(columns=['TimeToDown', 'Timestamp']).iloc[i:i + sequence_length].values
    X_train.append(scaler.fit_transform(seq))
    y_train.append(train_df['TimeToDown'].iloc[i + sequence_length - 1])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

X_test, y_test = [], []

for i in range(len(test_df) - sequence_length):
    seq = test_df.drop(columns=['TimeToDown', 'Timestamp']).iloc[i:i + sequence_length].values
    X_test.append(scaler.transform(seq))
    y_test.append(test_df['TimeToDown'].iloc[i + sequence_length - 1])

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# Enhanced LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[2]
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    rmse = torch.sqrt(criterion(y_pred, y_test))
    print(f'RMSE (log scale): {rmse.item():.4f}')

    y_pred_unscaled = np.expm1(y_pred.numpy())
    y_test_unscaled = np.expm1(y_test.numpy())
    rmse_unscaled = np.sqrt(np.mean((y_pred_unscaled - y_test_unscaled) ** 2))
    print(f'RMSE (minutes): {rmse_unscaled:.4f}')

# Predict next Down
columns_to_drop = ['TimeToDown', 'Timestamp']
if 'FirstOutOfRange' in test_df.columns:
    columns_to_drop.append('FirstOutOfRange')
last_seq = test_df.drop(columns=columns_to_drop).iloc[-sequence_length:].values
last_seq_scaled = scaler.transform(last_seq)
last_seq_tensor = torch.FloatTensor(last_seq_scaled).unsqueeze(0)
pred_lag_log = model(last_seq_tensor).item()
pred_lag = np.expm1(pred_lag_log)
last_time = test_df['Timestamp'].iloc[-1]
next_down = last_time + pd.Timedelta(minutes=pred_lag)
print(f"Predicted next 'Down' status at: {next_down} (lag: {pred_lag:.2f} minutes)")
