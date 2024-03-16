import os
import numpy as np

# Define the base path where the data is stored
DATA_PATH = '/home/targol/EpilepticSeizur/data/chb01'

# List all .npy files in the data directory
data_files = [file for file in os.listdir(DATA_PATH) if file.endswith('.npy')]

# Check existence
if not data_files:
    print("No data files found.")
else:
    print(f"Total data files found: {len(data_files)}")

# Load a few segments for data integrity check
sample_files = data_files[:5]  # Check the first 5 files
for file in sample_files:
    file_path = os.path.join(DATA_PATH, file)
    data = np.load(file_path)
    print(f"File: {file}, Shape: {data.shape}")

    # Statistical summary for the first EEG channel
    if data.ndim == 2:  # Ensure data is 2D (channels x samples)
        first_channel_data = data[0, :]
        mean_val = np.mean(first_channel_data)
        std_dev = np.std(first_channel_data)
        min_val = np.min(first_channel_data)
        max_val = np.max(first_channel_data)
        print(f"Statistics for the first channel of {file}:")
        print(f"Mean: {mean_val:.2f}, Std Dev: {std_dev:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}\n")
