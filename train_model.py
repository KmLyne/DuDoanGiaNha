import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Select only the RM, PTRATIO, and LSTAT features
# Adjusted column indices for RM, PTRATIO, and LSTAT
selected_features = data[:, [5, 10, 12]]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model has been trained and saved as model.pkl")
