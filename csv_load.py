import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import json
import sys
import numpy as np
import os

# Import tinyphysics components
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, BaseController

torch.backends.cudnn.enabled = False

print(torch.cuda.is_available())

def check_gpu():
    if torch.cuda.is_available():
        # Use the default CUDA device (usually 0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    else:
        print("CUDA error")
        print("Unable to use GPU. Exiting.")
        sys.exit(1)

# Force GPU usage
device = check_gpu()
print(f"Using device: {device}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def csv_generator(files):
    for file in tqdm(files, desc="Reading CSV files", unit="file"):
        yield pd.read_csv(file)

def load_and_prepare_data(file_pattern):
    print("Step 1: Data Preparation")
    all_files = glob.glob(file_pattern)
    features = ['vEgo', 'aEgo', 'roll', 'steerCommand']
    target = 'targetLateralAcceleration'
    
    print(f"Found {len(all_files)} CSV files")
    df = pd.concat(csv_generator(all_files), ignore_index=True)
    
    print(f"Total rows in concatenated DataFrame: {df.shape[0]}")
    unified_data = pd.DataFrame()
    scaler = StandardScaler()
    chunk_size = 10000

    print("Processing data in chunks...")
    for file in all_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            if chunk.empty or chunk[features].empty:
                print(f"Warning: Empty chunk or no features found in file {file}")
                continue

            chunk.loc[:, features] = scaler.fit_transform(chunk[features])
            unified_data = pd.concat([unified_data, chunk[features + [target]]], ignore_index=True)

    print("Calculating PID gains...")
    unified_data['Kp'] = unified_data[target] * 0.1
    unified_data['Ki'] = unified_data[target] * 0.01
    unified_data['Kd'] = unified_data[target] * 0.001

    X = torch.tensor(unified_data[features].values, dtype=torch.float32).to(device)
    y = torch.tensor(unified_data[['Kp', 'Ki', 'Kd']].values, dtype=torch.float32).to(device)
    print("Splitting data into train and validation sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        lstm_out, _ = self.lstm(x)
        if lstm_out.dim() == 2:
            return self.fc(lstm_out)
        else:
            return self.fc(lstm_out[:, -1, :])

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, patience):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        epoch_loss = 0
        batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_train_loss = epoch_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    return train_losses, val_losses, val_pred

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.close()

def test_model_with_tinyphysics(model, X_val, y_val):
    class ModelController(BaseController):
        def __init__(self, model):
            self.model = model

        def update(self, target_lataccel, current_lataccel, state, future_plan):
            input_data = torch.tensor([state.v_ego, state.a_ego, state.roll_lataccel, 0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                gains = self.model(input_data)
            return gains[0, 0].item()  # Return Kp as the steering command

    tiny_model = TinyPhysicsModel('path_to_tiny_physics_model.onnx', debug=False)
    controller = ModelController(model)
    simulator = TinyPhysicsSimulator(tiny_model, 'path_to_test_data.csv', controller=controller, debug=False)
    results = simulator.rollout()
    return results

def load_preprocess_and_save_data(file_pattern, output_file):
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val = load_and_prepare_data(file_pattern)
    
    # Convert to numpy arrays and save
    np.savez(output_file,
             X_train=X_train.cpu().numpy(),
             X_val=X_val.cpu().numpy(),
             y_train=y_train.cpu().numpy(),
             y_val=y_val.cpu().numpy())
    
    print(f"Preprocessed data saved to {output_file}")

def load_preprocessed_data(input_file, device):
    print(f"Loading preprocessed data from {input_file}")
    data = np.load(input_file)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32).to(device)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32).to(device)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32).to(device)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32).to(device)
    return X_train, X_val, y_train, y_val

def main():
    print("Starting main workflow...")
    
    preprocessed_data_file = 'preprocessed_data.npz'
    
    # Check if preprocessed data exists
    if not os.path.exists(preprocessed_data_file):
        load_preprocess_and_save_data('D:\Github\controls_challenge\data\*.csv', preprocessed_data_file)
    
    # Load preprocessed data
    X_train, X_val, y_train, y_val = load_preprocessed_data(preprocessed_data_file, device)

    print("Initializing and training the model...")
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=64, output_size=y_train.shape[1]).to(device)
    train_losses, val_losses, val_pred = train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=1000, patience=10)

    plot_losses(train_losses, val_losses)

    val_pred_np = val_pred.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    mae = mean_absolute_error(y_val_np, val_pred_np)
    r2 = r2_score(y_val_np, val_pred_np)
    print(f"MAE: {mae}, R2 Score: {r2}")

    print("Testing model with TinyPhysics...")
    tinyphysics_results = test_model_with_tinyphysics(model, X_val, y_val)
    print("TinyPhysics Results:", tinyphysics_results)

    # Save model information
    model_info = {
        "architecture": "LSTM",
        "input_size": X_train.shape[1],
        "hidden_size": 64,
        "output_size": y_train.shape[1],
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "mae": mae,
        "r2_score": r2,
        "tinyphysics_results": tinyphysics_results
    }

    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)

    print("Workflow completed!")

if __name__ == "__main__":
    main()