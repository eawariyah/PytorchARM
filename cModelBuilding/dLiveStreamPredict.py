import torch
import joblib
import torch.nn as nn
import numpy as np
import serial
import re

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

# Configure the serial connection (adjust parameters as necessary)
ser = serial.Serial('COM3', 115200, timeout=1)  # Adjust for Windows
# ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust for Linux

# Compile a regex pattern to match boot log lines you want to exclude
exclude_pattern = re.compile(
    r"ets Jul 29 2019 \d{2}:\d{2}:\d{2}|rst:0x1 \(POWERON_RESET\),boot:0x13 \(SPI_FAST_FLASH_BOOT\)|configsip: 0, SPIWP:0xee|clk_drv:0x00,q_drv:0x00,d_drv:0x00,cs0_drv:0x00,hd_drv:0x00,wp_drv:0x00|mode:DIO, clock div:1|load:0x3fff0030,len:1344|load:0x40078000,len:13964|load:0x40080400,len:3600|entry 0x400805f0"
)

# message = 'Ready'
# ser.write(message.encode())
# print(f"Sent: {message}")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # Check if the line matches the exclude pattern
            if not exclude_pattern.match(line):
                # Split the data into individual values
                LiveData = line.split(',')

                # Ensure the LiveData has exactly 3 values (Temperature, Pressure, Altitude)
                if len(LiveData) == 3:
                    try:
                        BME280Reading = [float(x) for x in LiveData]
                        # print(f"Received data: {BME280Reading}")

                        # Preprocess the test data (scaling)
                        test = np.array(BME280Reading).reshape(1, -1)
                        test = scaler.transform(test)

                        # Convert to torch tensor
                        test_tensor = torch.tensor(test, dtype=torch.float32)

                        # Make prediction
                        with torch.no_grad():
                            prediction = model(test_tensor)
                            rain_probability = prediction.item() * 100  # Convert to percentage

                        print(f"Predicted probability of rain: {rain_probability:.2f}%")
                    except ValueError:
                        print("Error: Received invalid data. Could not convert to float.")
                else:
                    print("Error: Received incorrect number of data points.")
    except KeyboardInterrupt:
        print("Exiting program")
        break
    except Exception as e:
        print(f"Error: {e}")

ser.close()
