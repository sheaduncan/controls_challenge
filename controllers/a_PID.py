import numpy as np
from controllers import BaseController
from collections import namedtuple, deque

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

class PIDLearner:
  def __init__(self):
      self.base_gains = {
          'low_speed': {'kp': 0.8, 'ki': 0.1, 'kd': 0.3},
          'medium_speed': {'kp': 0.7, 'ki': 0.08, 'kd': 0.25},
          'high_speed': {'kp': 0.6, 'ki': 0.06, 'kd': 0.2}
      }
      self.current_gains = self.base_gains['medium_speed'].copy()
      self.last_error = 0
      self.integral = 0
      self.max_integral = 2.0
      
      # Adaptive learning parameters
      self.base_lr = 0.05  # Add this line
      self.adaptive_lr = self.base_lr
      self.lr_decay = 0.999
      self.lr_min = 0.005
      self.momentum = 0.9
      self.velocity = {'kp': 0, 'ki': 0, 'kd': 0}

      # Cost tracking
      self.cost_window = deque(maxlen=100)
      self.max_cost = 1.0  # This will be updated as we see costs
      
  def get_gains(self, v_ego, cost):
        base_gains = self.base_gains['medium_speed']  # Default to medium speed
        if v_ego < 10:
            base_gains = self.base_gains['low_speed']
        elif v_ego > 30:
            base_gains = self.base_gains['high_speed']
        
        cost_factor = min(cost / self.max_cost, 2.0)  # Cap at 2x
        return {k: v * (1 + 0.5 * cost_factor) for k, v in base_gains.items()}
  
  def calculate_pid(self, error, dt, future_plan):
      self.integral = np.clip(self.integral + error * dt, -self.max_integral, self.max_integral)
      
      # Calculate regular derivative
      derivative = (error - self.last_error) / dt
      
      # Safety check for future_plan
      future_lataccel = future_plan.lataccel[1] if len(future_plan.lataccel) > 1 else error
      
      # Calculate derivative kick
      derivative_kick = (future_lataccel - error) / dt
      
      # Increased and adaptive feedforward term
      feedforward_gain = 1.2 + 0.5 * abs(error)  # Increases with error magnitude
      feedforward = feedforward_gain * (future_lataccel - error)
      
      # More aggressive non-linear term
      non_linear_term = np.sign(error) * np.power(abs(error), 0.7) * 1.0
      
      output = (self.current_gains['kp'] * error + 
                self.current_gains['ki'] * self.integral + 
                self.current_gains['kd'] * derivative +
                self.current_gains['kd'] * derivative_kick +  # Add derivative kick
                feedforward +
                non_linear_term)
  
      # Anti-windup
      if abs(self.integral) > self.max_integral:
          self.integral = np.sign(self.integral) * self.max_integral
  
      self.last_error = error
      return output
  
  def calculate_cost(self, error, state, future_plan):
      lataccel_cost = error**2 * 100
      jerk = (state.roll_lataccel - self.last_error) / 0.1
      jerk_cost = jerk**2 * 5
      total_cost = lataccel_cost + jerk_cost
        
      # Update max cost seen
      self.max_cost = max(self.max_cost, total_cost)
        
      # Add to cost window
      self.cost_window.append(total_cost)
        
      return total_cost
  
  def update(self, error, state, future_plan, dt):
      cost = self.calculate_cost(error, state, future_plan)
        
        # Adapt learning rate based on recent costs
      avg_cost = np.mean(self.cost_window) if self.cost_window else 0
      self.adaptive_lr = self.base_lr * (1 + avg_cost / self.max_cost)
      self.adaptive_lr = max(self.lr_min, min(self.adaptive_lr, 0.1))  # Bound between lr_min and 0.1
        
      for gain in ['kp', 'ki', 'kd']:
          gradient = self.calculate_gradient(gain, error, state, future_plan, dt)
          self.velocity[gain] = self.momentum * self.velocity[gain] - self.adaptive_lr * gradient
          self.current_gains[gain] += self.velocity[gain]
          self.current_gains[gain] = np.maximum(0, self.current_gains[gain])
        
          # Adapt gains based on cost
          adapted_gains = self.get_gains(state.v_ego, cost)
      for gain in ['kp', 'ki', 'kd']:
          self.current_gains[gain] = 0.9 * self.current_gains[gain] + 0.1 * adapted_gains[gain]

  def calculate_gradient(self, gain, error, state, future_plan, dt):
      epsilon = 1e-6
      original_gain = self.current_gains[gain]
      
      current_cost = self.calculate_cost(error, state, future_plan)
      
      self.current_gains[gain] += epsilon
      new_output = self.calculate_pid(error, dt, future_plan)
      new_error = error - new_output
      new_cost = self.calculate_cost(new_error, state, future_plan)
      
      self.current_gains[gain] = original_gain
      
      gradient = (new_cost - current_cost) / epsilon
      return gradient

class Controller(BaseController):
  def __init__(self):
      self.pid_learner = PIDLearner()
      self.last_time = None
      self.last_output = 0
      self.smooth_threshold = 0.5
      self.smooth_factor = 0.5
      
  def update(self, target_lataccel, current_lataccel, state, future_plan):

      # Ensure inputs are scalar
      target_lataccel = target_lataccel.item() if hasattr(target_lataccel, 'shape') else target_lataccel
      current_lataccel = current_lataccel.item() if hasattr(current_lataccel, 'shape') else current_lataccel

      current_time = state.v_ego / 10
      if self.last_time is None:
          self.last_time = current_time
          return 0
      dt = current_time - self.last_time
      self.last_time = current_time
      

      error = target_lataccel - current_lataccel
      
      # Calculate cost
      cost = self.pid_learner.calculate_cost(error, state, future_plan)

      gains = self.pid_learner.get_gains(state.v_ego, cost)
      self.pid_learner.current_gains = gains
      
      # Ensure future_plan is a single FuturePlan object and has valid data
      if isinstance(future_plan, list):
          future_plan = future_plan[0] if future_plan else FuturePlan([], [], [], [])
      elif not isinstance(future_plan, FuturePlan):
          future_plan = FuturePlan([], [], [], [])
      
      raw_output = self.pid_learner.calculate_pid(error, dt, future_plan)
      
      raw_output = raw_output.item() if hasattr(raw_output, 'shape') else raw_output

      # Adaptive smoothing with dynamic threshold
      error_magnitude = abs(error)
      dynamic_threshold = self.smooth_threshold * (1 + error_magnitude)
      if abs(raw_output - self.last_output) < dynamic_threshold:
          smooth_factor = self.smooth_factor / (1 + error_magnitude)
          output = self.last_output + smooth_factor * (raw_output - self.last_output)
      else:
          output = raw_output
      
      # Dynamic rate limiting
      max_change = 2.0 * dt * (1 + error_magnitude)
      output = self.last_output + np.clip(output - self.last_output, -max_change, max_change)
      
      self.pid_learner.update(error, state, future_plan, dt)
      
      self.last_output = output
      return output