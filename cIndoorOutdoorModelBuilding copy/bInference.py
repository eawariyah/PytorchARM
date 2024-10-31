import torch
import joblib
import torch.nn as nn
import numpy as np

# Define the model structure for indoor/outdoor prediction
class IndoorOutdoorModel(nn.Module):
    def __init__(self):
        super(IndoorOutdoorModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Adjusted input size to 2 for Temperature and Humidity
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the trained model
model = IndoorOutdoorModel()
model.load_state_dict(torch.load('indoor_outdoor_model.pkl'))  # Load the appropriate model
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Test array representing [Temperature, Humidity]
test = [28.04, 65.5]  # Example temperature and humidity values

# Preprocess the test data (scaling)
test = np.array(test).reshape(1, -1)
test = scaler.transform(test)

# Convert to torch tensor
test_tensor = torch.tensor(test, dtype=torch.float32)

# Make prediction
with torch.no_grad():
    prediction = model(test_tensor)
    indoors_probability = prediction.item() * 100  # Convert to percentage

print(f"Predicted probability of being indoors: {indoors_probability:.2f}%")
