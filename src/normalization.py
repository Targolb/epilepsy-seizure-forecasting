import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the directory where your data is saved
data_dir = '/home/targol/EpilepticSeizur/data'

# Load the balanced datasets
pre_seizure_data = np.load(data_dir + '/balanced_pre_seizure.npy')
non_seizure_data = np.load(data_dir + '/balanced_non_seizure.npy')

# Flatten the data since StandardScaler works on 2D data
X_pre_seizure = pre_seizure_data.reshape(pre_seizure_data.shape[0], -1)
X_non_seizure = non_seizure_data.reshape(non_seizure_data.shape[0], -1)

# Combine pre-seizure and non-seizure data
X_combined = np.concatenate((X_pre_seizure, X_non_seizure), axis=0)

# Initialize the scaler
scaler = StandardScaler()

# Fit on the combined data and transform it
X_scaled = scaler.fit_transform(X_combined)

# Split the scaled data back into pre-seizure and non-seizure
X_pre_seizure_scaled = X_scaled[:X_pre_seizure.shape[0]]
X_non_seizure_scaled = X_scaled[X_pre_seizure.shape[0]:]

# Reshape the data back to its original shape
pre_seizure_data_scaled = X_pre_seizure_scaled.reshape(pre_seizure_data.shape)
non_seizure_data_scaled = X_non_seizure_scaled.reshape(non_seizure_data.shape)

# Save the scaled data
np.save(data_dir + '/pre_seizure_scaled.npy', pre_seizure_data_scaled)
np.save(data_dir + '/non_seizure_scaled.npy', non_seizure_data_scaled)

print(f"Pre-seizure scaled data shape: {pre_seizure_data_scaled.shape}")
print(f"Non-seizure scaled data shape: {non_seizure_data_scaled.shape}")
