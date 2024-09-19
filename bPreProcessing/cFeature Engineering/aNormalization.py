

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define file paths
RainyDayPath = 'aDataCollection/RainyDay.csv'
SunnyDayPath = 'aDataCollection/SunnyDay.csv'

# Read CSV files
RainyDayDF = pd.read_csv(RainyDayPath)
SunnyDayDF = pd.read_csv(SunnyDayPath)

# Combine both DataFrames
combinedDF = pd.concat([RainyDayDF, SunnyDayDF], ignore_index=True)

# Initialize StandardScaler
scaler = StandardScaler()

# Standardize numerical columns (Temperature, Pressure, Altitude)
standardized_values = scaler.fit_transform(combinedDF[['Temperature', 'Pressure', 'Altitude']])

# Create a new DataFrame for standardized data
standardizedDF = pd.DataFrame(standardized_values, columns=['Temperature', 'Pressure', 'Altitude'])

# Add the timestamp back to the standardized DataFrame
standardizedDF['Timestamp'] = combinedDF['Temperature'].index.map(lambda x: combinedDF['Temperature'][x])

# Separate back into Rainy and Sunny if needed
standardizedRainyDF = standardizedDF.iloc[:len(RainyDayDF)]
standardizedSunnyDF = standardizedDF.iloc[len(RainyDayDF):]

# Display standardized dataframes
print("Standardized Rainy Day Data:")
print(standardizedRainyDF)

print("\nStandardized Sunny Day Data:")
print(standardizedSunnyDF)