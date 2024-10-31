import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load datasets
RainyDayPath = 'aDataCollection/RainyDayNormal.csv'
SunnyDayPath = 'aDataCollection/SunnyDay.csv'

RainyDayDF = pd.read_csv(RainyDayPath)
SunnyDayDF = pd.read_csv(SunnyDayPath)

# Add a "Rain" column: 1 for rainy day, 0 for sunny day
RainyDayDF['Rain'] = 1
SunnyDayDF['Rain'] = 0

# Concatenate both dataframes
df = pd.concat([RainyDayDF, SunnyDayDF])

# Feature columns
X = df[['Temperature', 'Pressure', 'Altitude']].values
y = df['Rain'].values

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use in inference
joblib.dump(scaler, 'scaler.pkl')

# Define a simple feedforward neural network
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

# Initialize the model, loss function, and optimizer
model = RainPredictionModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Function to evaluate the model and calculate metrics
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = outputs.round()  # Convert sigmoid output to binary (0 or 1)
        
        y_test_np = y_test.numpy()
        predicted_np = predicted.numpy()
        
        accuracy = accuracy_score(y_test_np, predicted_np)
        precision = precision_score(y_test_np, predicted_np)
        recall = recall_score(y_test_np, predicted_np)
        f1 = f1_score(y_test_np, predicted_np)
        
        return accuracy, precision, recall, f1

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Every 10 epochs, calculate and print the metrics on the test dataset
    if (epoch+1) % 10 == 0:
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

