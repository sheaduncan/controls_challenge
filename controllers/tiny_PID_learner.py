import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from controllers import BaseController
from collections import deque
from tinygrad import Device

class ActiveModelLearner:
  def __init__(self, input_size=5, hidden_size=64, output_size=1):
      self.fc1 = Linear(input_size, hidden_size)
      self.fc2 = Linear(hidden_size, hidden_size)
      self.fc3 = Linear(hidden_size, output_size)

  def __call__(self, x):
      x = self.fc1(x).relu()
      x = self.fc2(x).relu()
      return self.fc3(x)

class Controller(BaseController):
  def __init__(self):
      # PID gains
      self.Kp = 0.6
      self.Ki = 0.005
      self.Kd = 0.12
      
      # Controller state
      self.error_sum = 0
      self.last_error = 0
      self.last_target = 0
      self.filtered_error = 0
      
      # Active Model Learner
      self.model = ActiveModelLearner()
      self.optimizer = Adam([self.model.fc1.weight, self.model.fc1.bias,
                             self.model.fc2.weight, self.model.fc2.bias,
                             self.model.fc3.weight, self.model.fc3.bias], lr=0.0005)
      
      # Data buffer for learning
      self.buffer_size = 1000
      self.ma_window = 10
      self.data_buffer = deque(maxlen=self.buffer_size)
      
      # Learning rate for online updates
      self.learning_rate = 0.05
      
      # Moving average for predictions
      self.prediction_history = deque(maxlen=self.ma_window)
      
      # Low-pass filter coefficient
      self.alpha = 0.2
      
      # Low-pass filter for control output
      self.u_filter = 0  # For low-pass filtering the control output
      self.filter_coeff = 0.2  # Adjust this to change filter strength (0-1)

      # Add these new parameters
      self.max_jerk = 0.2  # Maximum allowed jerk
      self.prev_steer_command = 0  # To track previous steering command
      self.steer_rate_limit = 0.1  # Maximum change in steering per update
      self.jerk_threshold = 0.6  # Threshold for considering a jerk "large"
      self.smooth_factor = 1.0  # Smoothing factor for small jerks (0-1)

      # Batch size for model updates
      self.batch_size = 32  # or any other appropriate value

  def update_model(self):
      if len(self.data_buffer) < self.batch_size:
          return
      
      batch_indices = np.random.choice(len(self.data_buffer), self.batch_size, replace=False)
      batch = [self.data_buffer[i] for i in batch_indices]
      
      inputs = Tensor([b[0] for b in batch], device='GPU')
      targets = Tensor([[b[1]] for b in batch], device='GPU')
      
      Tensor.training = True  # Set training mode to True
      outputs = self.model(inputs)
      loss = ((outputs - targets)**2).mean()
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      Tensor.training = False  # Set training mode back to False

  def predict_lataccel(self, v_ego, a_ego, roll, steer_command, target_lataccel):
      input_data = Tensor([[v_ego, a_ego, roll, steer_command, target_lataccel]], device='GPU')
      
      # Manually disable gradient computation for prediction
      prev_training = Tensor.training
      Tensor.training = False
      prediction = self.model(input_data).numpy().item()
      Tensor.training = prev_training
          
      self.prediction_history.append(prediction)
      
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
      
      # Add a check for the future_plan
      if future_plan and len(future_plan) > 0 and len(future_plan[0]) > 1:
          model_error = future_plan[0][1] - predicted_lataccel
      else:
          model_error = 0  # or some other default value
      
      u_model = self.learning_rate * model_error
      
      u = u_pid + u_model
      
      # Calculate the proposed change in steering
      delta_u = u - self.prev_steer_command
      
      # Apply smoothing only to small jerks
      if abs(delta_u) < self.jerk_threshold:
          u = self.prev_steer_command + self.smooth_factor * delta_u
      else:
          # For large jerks, allow the full change
          u = u
      
      # Update previous steering command
      self.prev_steer_command = u
      
      # Apply low-pass filter
      self.u_filter = self.filter_coeff * u + (1 - self.filter_coeff) * self.u_filter
      u = self.u_filter
      
      # Apply jerk limiting
      max_change = self.max_jerk * (1 / 20)  # Assuming 20Hz update rate
      u = np.clip(u, self.prev_steer_command - max_change, self.prev_steer_command + max_change)

      # Apply steering rate limiting
      u = np.clip(u, self.prev_steer_command - self.steer_rate_limit, self.prev_steer_command + self.steer_rate_limit)

      # Update previous steering command
      self.prev_steer_command = u
      
      input_data = [v_ego, a_ego, roll, u, target_lataccel]
      self.data_buffer.append((input_data, current_lataccel))
      
      self.update_model()
      
      self.last_error = self.filtered_error
      self.last_target = target_lataccel
      
      return np.clip(u, -2, 2)

Device.DEFAULT = 'GPU'