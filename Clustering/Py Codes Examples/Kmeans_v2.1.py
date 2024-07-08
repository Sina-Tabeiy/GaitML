import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Directory containing the CSV files
data_dir = r'G:\Uni\PhD\K-means\Python code\Example Dataset\Test'

# Load the data
data_list = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        df.drop('activity', axis= 1, inplace= True)
        data_list.append(df)

print("Number of participants:", len(data_list))
print("Sample data for first participant:\n", data_list[0].head())

# Define the number of points for time normalization
num_points = 101

# Function to time-normalize data to a specific number of points
def time_normalize(df, num_points):
    time_normalized_df = pd.DataFrame()
    for column in df.columns:
        time_normalized_df[column] = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(df)), df[column])

    return time_normalized_df

# Apply time normalization to each participant's data
normalized_data_list = [time_normalize(df, num_points) for df in data_list]

print("Time-normalized data for first participant:\n", normalized_data_list[0].head())


# Combine all participant data into a data matrix
data_matrix = np.array([df.values.flatten() for df in normalized_data_list])
#pd.DataFrame(data_matrix).to_excel("output.xlsx", index= False)
print("Data matrix shape:", data_matrix.shape)  # Should be (number of participants, 505)

# Optionally, save the clustered data
pd.DataFrame(data_matrix).to_csv('data_matrix.csv', index=False)

# Determine the optimal number of clusters (optional)
inertia = []
silhouette_scores = []
range_clusters = range(2, 4)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_matrix)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_matrix, kmeans.labels_))

# Plotting the Elbow method
plt.figure(figsize=(10, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Plotting the Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(range_clusters, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., 3)
optimal_k = 3  # Replace with the number you choose based on the above methods
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_matrix)

# Get the cluster labels
labels = kmeans.labels_

# Add cluster labels to the original data
result_df = pd.DataFrame(data_matrix, columns=[f'feature_{i}' for i in range(data_matrix.shape[1])])
result_df['Cluster'] = labels

print("Clustered data sample:\n", result_df.head())


#result_df.to_csv('clustered_gait_data.csv', index=False)
