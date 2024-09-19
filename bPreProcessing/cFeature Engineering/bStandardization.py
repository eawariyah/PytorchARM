import pandas as pd
from sklearn.preprocessing import StandardScaler

big_table_path = 'aDataCollection/eBigTable/BigTable.csv'
big_data = pd.read_csv(big_table_path)

timestamps = big_data['Timestamp']
big_data = big_data.drop('Timestamp', axis=1)

# Z-Score Normalization (Standardization)
standard_scaler = StandardScaler()
big_data_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(big_data), columns=big_data.columns)

# Add the Timestamp column back
big_data_standard_scaled['Timestamp'] = timestamps

# Save the Standardized data
standard_normalized_path = 'aDataCollection/eBigTable/BigTable_StandardNormalized.csv'
big_data_standard_scaled.to_csv(standard_normalized_path, index=False)
print(f"Standardized data written to {standard_normalized_path}")