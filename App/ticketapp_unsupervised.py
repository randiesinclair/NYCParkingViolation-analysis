import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
import numpy as np

def predict_ticket_probability(kmeans, scaler, encoders):
    # Collect user input
    user_input = []
    user_input.append(input("1. What is the make of the vehicle? "))
    user_input.append(input("2. What is the color of the vehicle? "))
    user_input.append(input("3. What is the registration state? "))
    user_input.append(input("4. What is the vehicle body type? "))
    user_input.append(input("5. What is the street name you're parked on? "))
    
    # Encode and scale the input
    user_input_encoded = [encoders[i].transform([user_input[i]])[0] for i in range(5)]
    user_input_scaled = scaler.transform([user_input_encoded])

    # Predict the cluster for the input data
    cluster = kmeans.predict(user_input_scaled)[0]
    
    # Compute the probability based on the cluster size
    cluster_size = len(data_sample[data_sample['Cluster'] == cluster])
    total_records = len(data_sample)
    probability = cluster_size / total_records
    
    return probability

# Load the data
data = pd.read_csv("C:/Users/Owner/Desktop/Code Repos/NYCParkingViolation-analysis/NYCParkingViolations_FiscalYear_2023.csv")

# Preprocess the data
data = data.sample(frac=0.1, random_state=42)
data_sample = data[['Vehicle Make', 'Vehicle Color', 'Registration State', 'Vehicle Body Type', 'Street Name']]
encoders = [LabelEncoder() for _ in range(5)]
data_encoded = pd.DataFrame()

for i, column in enumerate(data_sample.columns):
    data_encoded[column] = encoders[i].fit_transform(data_sample[column])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Train the k-means clustering model
n_clusters = 5
with parallel_backend('threading', n_jobs=-1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)

data_sample['Cluster'] = kmeans.labels_

# Compute the accuracy of the clustering model
accuracy = accuracy_score(data_sample['Cluster'], kmeans.labels_)
print(f"\nThe accuracy of the clustering model is {accuracy * 100:.2f}%")

# Predict the probability of getting a ticket
probability = predict_ticket_probability(kmeans, scaler, encoders)
print(f"\nThe probability of getting a ticket is {probability * 100:.2f}%.")
