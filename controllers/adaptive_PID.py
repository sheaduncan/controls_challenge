import numpy as np
from controllers import BaseController

class Controller(BaseController):
  def __init__(self):
      # Initial PID gains
      self.Kp = 0.6  # Reduced from 1.0
      self.Ki = 0.05  # Reduced from 0.1
      self.Kd = 0.1  # Increased from 0.05
      
      # Adaptive parameters
      self.alpha = 0.005  # Reduced learning rate
      self.error_sum = 0
      self.last_error = 0
      self.filtered_error = 0
      self.error_history = []
      
      # Limits for gain adaptation
      self.Kp_limits = (0.1, 1.0)
      self.Ki_limits = (0.01, 0.1)
      self.Kd_limits = (0.05, 0.2)

      # Low-pass filter coefficient
      self.filter_coeff = 0.1

  def adapt_gains(self, error, v_ego):
      # Gain scheduling based on error magnitude
      error_mag = abs(error)
      if error_mag > 1.0:
          self.Kp *= 0.9
          self.Ki *= 0.9
          self.Kd *= 1.1
      elif error_mag < 0.1:
          self.Kp *= 1.1
          self.Ki *= 1.1
          self.Kd *= 0.9

      # Clip gains to prevent instability
      self.Kp = np.clip(self.Kp, *self.Kp_limits)
      self.Ki = np.clip(self.Ki, *self.Ki_limits)
      self.Kd = np.clip(self.Kd, *self.Kd_limits)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = target_lataccel - current_lataccel
      
      # Low-pass filter on error
      self.filtered_error = self.filter_coeff * error + (1 - self.filter_coeff) * self.filtered_error

      self.error_sum += self.filtered_error
      error_diff = self.filtered_error - self.last_error

      # Adapt gains
      self.adapt_gains(self.filtered_error, state.v_ego)

      # Calculate control action
      u = (self.Kp * self.filtered_error + 
           self.Ki * self.error_sum + 
           self.Kd * error_diff)

      # Feedforward term
      u += 0.4 * target_lataccel

      # Anti-windup for integral term
      if abs(u) >= 2:
          self.error_sum -= self.filtered_error

      # Update error history
      self.error_history.append(self.filtered_error)
      if len(self.error_history) > 100:
          self.error_history.pop(0)

      self.last_error = self.filtered_error

      return np.clip(u, -2, 2)  # Clip to steer range