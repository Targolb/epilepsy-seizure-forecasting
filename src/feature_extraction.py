import numpy as np
import scipy.signal
import os

# Load normalized data
data_dir = '/home/targol/EpilepticSeizur/data'
pre_seizure_data_scaled = np.load(os.path.join(data_dir, 'pre_seizure_scaled.npy'))
non_seizure_data_scaled = np.load(os.path.join(data_dir, 'non_seizure_scaled.npy'))

# STFT parameters
fs = 256  # Sampling frequency (update accordingly)
window = 'hann'
nperseg = 128  # Number of samples per STFT segment
noverlap = nperseg // 2  # Overlap between STFT segments


def compute_mean_psd(data):
    mean_psds = []
    for segment in data:
        # Assume each segment is already 1-dimensional
        # Perform STFT directly on the segment
        _, _, Zxx = scipy.signal.stft(segment, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        PSD = np.abs(Zxx) ** 2
        mean_PSD = np.mean(PSD, axis=1)
        mean_psds.append(mean_PSD)
    return np.array(mean_psds)


# Compute mean PSD for pre-seizure and non-seizure data
mean_psd_pre_seizure = compute_mean_psd(pre_seizure_data_scaled)
mean_psd_non_seizure = compute_mean_psd(non_seizure_data_scaled)

# Save the mean PSD features
np.save(os.path.join(data_dir, 'mean_psd_pre_seizure.npy'), mean_psd_pre_seizure)
np.save(os.path.join(data_dir, 'mean_psd_non_seizure.npy'), mean_psd_non_seizure)
