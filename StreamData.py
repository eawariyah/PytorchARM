import serial
import json
import time
import pandas as pd
from datetime import datetime

def is_valid_data(data):
    expected_keys = ["L410", "L435", "L460", "L485", "L510", "L535", "L560", "L585", "L610", "L645", "L680", "L705", "L730", "L760", "L810", "L860", "L900", "L940"]
    
    # Check if all expected keys are present and their values are of type float
    for key in expected_keys:
        if key not in data or not isinstance(data[key], (int, float)):
            return False
    return True

# Initialize serial connections
# Microcontroller = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
Microcontroller = serial.Serial('COM3', 115200, timeout=1)

time.sleep(2)  # Wait for the serial connections to initialize

# Create a DataFrame to store the data
df = pd.DataFrame(columns=["timestamp"] + [f"L{wavelength}" for wavelength in ["410", "435", "460", "485", "510", "535", "560", "585", "610", "645", "680", "705", "730", "760", "810", "860", "900", "940"]])

while True:
    # Read JSON data from Microcontroller
    if Microcontroller.in_waiting > 0:
        received_data = Microcontroller.readline().decode('utf-8').strip()
        try:
            json_data = json.loads(received_data)
            print("Received from Microcontroller:", json_data)

            # Validate data types
            if is_valid_data(json_data):
                # Add timestamp to the data
                json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Convert json_data to DataFrame
                new_df = pd.DataFrame([json_data])

                # Check for empty or all-NA entries and exclude them
                if not new_df.isnull().all().all() and not new_df.empty:
                    # Concatenate the new DataFrame with the existing one
                    df = pd.concat([df, new_df], ignore_index=True)

                    # Save the DataFrame to a CSV file
                    df.to_csv('sensor_data.csv', index=False)
                else:
                    print("Empty or all-NA data received, not added to the DataFrame.")

            else:
                print("Invalid data received:", json_data)

        except json.JSONDecodeError:
            print("Failed to decode JSON from Microcontroller:", received_data)
