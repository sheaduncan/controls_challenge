import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import Adam
from controllers import BaseController
from collections import deque

class LearningModel:
    def __init__(self, input_size=8, hidden_size=128, output_size=1):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, hidden_size)
        self.fc4 = Linear(hidden_size, output_size)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        return self.fc4(x)

class Controller(BaseController):
    def __init__(self):
        # PID gains
        self.Kp = 0.3
        self.Ki = 0.005
        self.Kd = 0.05
        
        # Controller state
        self.error_sum = 0
        self.last_error = 0
        
        # Feedforward
        self.ff_horizon = 3
        self.ff_weight = 0.3
        
        # Learning Model
        self.model = LearningModel()
        self.optimizer = Adam([self.model.fc1.weight, self.model.fc1.bias,
                               self.model.fc2.weight, self.model.fc2.bias,
                               self.model.fc3.weight, self.model.fc3.bias], lr=0.004)
        
        # Data buffer for learning
        self.buffer_size = 1000
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        # Threshold for switching to model-based control
        self.model_threshold = 100
        
        # Batch size for model updates
        self.batch_size = 20

        self.last_prediction = 0
        self.smoothing_factor = 0.8  # Adjust this value between 0 and 1

    def calculate_feedforward(self, future_plan):
        future_lataccels = []
        for plan in future_plan[:self.ff_horizon]:
            if isinstance(plan, (list, tuple)) and len(plan) > 1:
                lataccel = plan[1]
                if not np.isnan(lataccel):
                    future_lataccels.append(lataccel)
        
        if not future_lataccels:
            return 0.0
        
        if len(future_lataccels) < self.ff_horizon:
            future_lataccels += [future_lataccels[-1]] * (self.ff_horizon - len(future_lataccels))
        
        weights = np.linspace(1, 0, self.ff_horizon)
        return np.average(future_lataccels, weights=weights)

    def smooth_prediction(self, new_prediction):
        threshold = 0.5  # Adjust this value to define "big" jumps
        if abs(new_prediction - self.last_prediction) > threshold:
            smoothed = new_prediction
        else:
            smoothed = self.smoothing_factor * self.last_prediction + (1 - self.smoothing_factor) * new_prediction
        self.last_prediction = smoothed
        return smoothed

    def update_model(self):
        if len(self.data_buffer) < self.batch_size:
            return
        
        batch_indices = np.random.choice(len(self.data_buffer), self.batch_size, replace=False)
        batch = [self.data_buffer[i] for i in batch_indices]
        
        inputs = Tensor([b[0] for b in batch])
        targets = Tensor([[b[1]] for b in batch])
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = ((outputs - targets)**2).mean()
        loss.backward()
        
        # Add this line before calling optimizer.step()
        Tensor.training = True
        
        self.optimizer.step()
        
        # Optionally, set it back to False after optimization if needed
        Tensor.training = False

    def predict_gain(self, v_ego, a_ego, roll, target_lataccel, future_plan, error):
        ff_signal = self.calculate_feedforward(future_plan)
        input_data = Tensor([[v_ego, a_ego, roll, target_lataccel, ff_signal, error, self.error_sum, error - self.last_error]])
        return self.model(input_data).detach().numpy().item()

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll = np.sin(state.roll_lataccel / 9.81)
        
        error = target_lataccel - current_lataccel
        self.error_sum += error
        error_diff = error - self.last_error
        
        ff_signal = self.calculate_feedforward(future_plan)
        
        if len(self.data_buffer) < self.model_threshold:
            # Use PID + Feedforward
            u = (self.Kp * error + 
                 self.Ki * self.error_sum + 
                 self.Kd * error_diff +
                 self.ff_weight * ff_signal)
        else:
            # Use Learning Model
            raw_prediction = self.predict_gain(v_ego, a_ego, roll, target_lataccel, future_plan, error)
            print("Raw Prediction:", raw_prediction)
            u = self.smooth_prediction(raw_prediction)
        
        # Store data for learning
        input_data = [v_ego, a_ego, roll, target_lataccel, ff_signal, error, self.error_sum, error_diff]
        self.data_buffer.append((input_data, u))
        
        # Update the model
        self.update_model()
        
        self.last_error = error
        
        return np.clip(u, -2, 2)