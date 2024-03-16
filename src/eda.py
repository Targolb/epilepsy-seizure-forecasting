import numpy as np
import matplotlib.pyplot as plt
import os

# Load the balanced datasets
pre_seizure_data = np.load('/home/targol/EpilepticSeizur/data/balanced_pre_seizure.npy')
non_seizure_data = np.load('/home/targol/EpilepticSeizur/data/balanced_non_seizure.npy')

# Dimensionality check
print(f"Pre-seizure data shape: {pre_seizure_data.shape}")
print(f"Non-seizure data shape: {non_seizure_data.shape}")

# Ensure the plots directory exists
plots_dir = '/home/targol/EpilepticSeizur/data/plots'
os.makedirs(plots_dir, exist_ok=True)

# Visual inspection of a few segments
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].plot(pre_seizure_data[0][0])  # Adjust indexing based on your data structure
axs[0, 0].set_title('Pre-Seizure Segment 1')
axs[0, 1].plot(pre_seizure_data[1][0])  # Adjust indexing
axs[0, 1].set_title('Pre-Seizure Segment 2')
axs[1, 0].plot(non_seizure_data[0][0])  # Adjust indexing
axs[1, 0].set_title('Non-Seizure Segment 1')
axs[1, 1].plot(non_seizure_data[1][0])  # Adjust indexing
axs[1, 1].set_title('Non-Seizure Segment 2')
plt.savefig(os.path.join(plots_dir, 'segment_visualization.png'))

# Distribution analysis (example)
plt.figure(figsize=(10, 6))
plt.hist(pre_seizure_data.flatten(), bins=50, alpha=0.5, label='Pre-Seizure')
plt.hist(non_seizure_data.flatten(), bins=50, alpha=0.5, label='Non-Seizure')
plt.legend()
plt.title('Value Distribution')
plt.savefig(os.path.join(plots_dir, 'value_distribution.png'))
