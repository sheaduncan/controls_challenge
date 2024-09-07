import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from controllers import BaseController

class ActiveModelLearner(nn.Module):
  def __init__(self, input_size=5, hidden_size=64, output_size=1):
      super(ActiveModelLearner, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, output_size)
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      return self.fc3(x)

class Controller(BaseController):
  def __init__(self):
      # PID gains
      self.Kp = 0.5
      self.Ki = 0.01
      self.Kd = 0.1
      
      # Controller state
      self.error_sum = 0
      self.last_error = 0
      self.last_target = 0
      self.filtered_error = 0
      
      # Active Model Learner
      self.model = ActiveModelLearner()
      self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
      self.loss_fn = nn.MSELoss()
      
      # Data buffer for learning
      self.data_buffer = []
      self.buffer_size = 2000
      self.batch_size = 64
      
      # Learning rate for online updates
      self.learning_rate = 0.05
      
      # Moving average for predictions
      self.prediction_history = []
      self.ma_window = 5
      
      # Low-pass filter coefficient
      self.alpha = 0.2

  def update_model(self):
      if len(self.data_buffer) < self.batch_size:
          return
      
      batch_indices = np.random.choice(len(self.data_buffer), self.batch_size, replace=False)
      batch = [self.data_buffer[i] for i in batch_indices]
      
      inputs = torch.FloatTensor([b[0] for b in batch])
      targets = torch.FloatTensor([b[1] for b in batch]).unsqueeze(1)
      
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.loss_fn(outputs, targets)
      loss.backward()
      self.optimizer.step()

  def predict_lataccel(self, v_ego, a_ego, roll, steer_command, target_lataccel):
      with torch.no_grad():
          input_data = torch.FloatTensor([[v_ego, a_ego, roll, steer_command, target_lataccel]])
          prediction = self.model(input_data).item()
          
      self.prediction_history.append(prediction)
      if len(self.prediction_history) > self.ma_window:
          self.prediction_history.pop(0)
      
      return np.mean(self.prediction_history)

  def adapt_gains(self, error):
      error_mag = abs(error)
      if error_mag > 0.5:
          self.Kp *= 0.99
          self.Ki *= 0.99
          self.Kd *= 1.01
      elif error_mag < 0.1:
          self.Kp *= 1.01
          self.Ki *= 1.01
          self.Kd *= 0.99
      
      self.Kp = np.clip(self.Kp, 0.1, 1.0)
      self.Ki = np.clip(self.Ki, 0.005, 0.05)
      self.Kd = np.clip(self.Kd, 0.05, 0.2)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      v_ego = state.v_ego
      a_ego = state.a_ego
      roll = np.sin(state.roll_lataccel / 9.81)
      
      error = target_lataccel - current_lataccel
      self.filtered_error = self.alpha * error + (1 - self.alpha) * self.filtered_error
      
      self.error_sum += self.filtered_error
      error_diff = self.filtered_error - self.last_error
      
      self.adapt_gains(self.filtered_error)
      
      u_pid = (self.Kp * self.filtered_error + 
               self.Ki * self.error_sum + 
               self.Kd * error_diff)
      
      predicted_lataccel = self.predict_lataccel(v_ego, a_ego, roll, u_pid, target_lataccel)
      model_error = target_lataccel - predicted_lataccel
      u_model = self.learning_rate * model_error
      
      u = u_pid + u_model
      
      input_data = [v_ego, a_ego, roll, u, target_lataccel]
      self.data_buffer.append((input_data, current_lataccel))
      if len(self.data_buffer) > self.buffer_size:
          self.data_buffer.pop(0)
      
      self.update_model()
      
      self.last_error = self.filtered_error
      self.last_target = target_lataccel
      
      return np.clip(u, -2, 2)