import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = '/home/targol/EpilepticSeizur/data'
plots_dir = os.path.join(data_dir, 'plots')

# Ensure the plots directory exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the mean PSD features
mean_psd_pre_seizure = np.load(os.path.join(data_dir, 'mean_psd_pre_seizure.npy'))
mean_psd_non_seizure = np.load(os.path.join(data_dir, 'mean_psd_non_seizure.npy'))

# Check shapes
print(f"Shape of mean_psd_pre_seizure: {mean_psd_pre_seizure.shape}")
print(f"Shape of mean_psd_non_seizure: {mean_psd_non_seizure.shape}")

# Check for NaN or infinite values
print("NaNs in pre-seizure:", np.isnan(mean_psd_pre_seizure).any())
print("Infs in pre-seizure:", np.isinf(mean_psd_pre_seizure).any())
print("NaNs in non-seizure:", np.isnan(mean_psd_non_seizure).any())
print("Infs in non-seizure:", np.isinf(mean_psd_non_seizure).any())

# Visualize the mean PSDs of a few segments from each category
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i in range(2):
    axs[0, i].plot(mean_psd_pre_seizure[i])
    axs[0, i].set_title(f'Pre-Seizure Segment {i + 1} Mean PSD')
    axs[1, i].plot(mean_psd_non_seizure[i])
    axs[1, i].set_title(f'Non-Seizure Segment {i + 1} Mean PSD')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'psd_check.png'))  # Save the figure to the plots directory
plt.show()
