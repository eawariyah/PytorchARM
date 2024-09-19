import serial
import csv
import datetime

# Configure the serial connection (the parameters might need adjusting)
# ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
ser = serial.Serial('COM3', 115200, timeout=1)


# Create or open the CSV file
with open('RainyDay.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Timestamp',"Temperature", "Pressure", "Altitude"])
    
    while True:
        try:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Split the data into individual values
                data = line.split(',')
                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # Add the timestamp to the data
                data.insert(0, timestamp)
                # Write the data to the CSV file
                writer.writerow(data)
                file.flush()  # Flush the buffer to ensure data is written to file
                print(data)
        except KeyboardInterrupt:
            print("Exiting program")
            break
        except Exception as e:
            print(f"Error: {e}")

ser.close()
