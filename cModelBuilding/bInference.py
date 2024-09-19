import torch
import joblib
import torch.nn as nn
import numpy as np

# Define the same model structure as used in TrainingModel.py
class RainPredictionModel(nn.Module):
    def __init__(self):
        super(RainPredictionModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the trained model
model = RainPredictionModel()
model.load_state_dict(torch.load('rain_prediction_model.pkl'))
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Test array representing [Temperature, Pressure, Altitude]
test = [28.04,98623.05,227.42]

# Preprocess the test data (scaling)
test = np.array(test).reshape(1, -1)
test = scaler.transform(test)

# Convert to torch tensor
test_tensor = torch.tensor(test, dtype=torch.float32)

# Make prediction
with torch.no_grad():
    prediction = model(test_tensor)
    rain_probability = prediction.item() * 100  # Convert to percentage

print(f"Predicted probability of rain: {rain_probability:.2f}%")
