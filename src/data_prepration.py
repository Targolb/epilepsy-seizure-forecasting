import numpy as np
import glob
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assuming your pre-seizure and non-seizure data are distinguishable by their filenames
data_dir = '/home/targol/EpilepticSeizur/data'
pre_seizure_files = glob.glob(data_dir + '/*pre_seizure*.npy')
non_seizure_files = glob.glob(data_dir + '/*non_seizure*.npy')


# Function to pad arrays to the maximum length found in the dataset
def pad_array(array, max_length):
    padding_size = max_length - array.shape[1]
    if padding_size > 0:
        return np.pad(array, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
    return array[:, :max_length]


# Find the maximum size among all files to standardize segment size
max_size = max([np.load(file).shape[1] for file in pre_seizure_files + non_seizure_files])

# Load, pad, and concatenate data
pre_seizure_data = np.concatenate([pad_array(np.load(file), max_size) for file in pre_seizure_files], axis=0)
non_seizure_data = np.concatenate([pad_array(np.load(file), max_size) for file in non_seizure_files], axis=0)

# Flatten the data since SMOTE works on 2D data
X_pre_seizure = pre_seizure_data.reshape(pre_seizure_data.shape[0], -1)
X_non_seizure = non_seizure_data.reshape(non_seizure_data.shape[0], -1)

X = np.concatenate((X_pre_seizure, X_non_seizure), axis=0)
y = np.array([1] * X_pre_seizure.shape[0] + [0] * X_non_seizure.shape[0])

# Apply SMOTE for balancing
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Reshape back to original data shape (excluding the length dimension)
original_shape = pre_seizure_data.shape[2:]  # Adjust according to your data's original shape
X_res_pre_seizure = X_res[y_res == 1].reshape((-1, max_size, *original_shape))
X_res_non_seizure = X_res[y_res == 0].reshape((-1, max_size, *original_shape))

# Save the balanced datasets
np.save(data_dir + '/balanced_pre_seizure.npy', X_res_pre_seizure)
np.save(data_dir + '/balanced_non_seizure.npy', X_res_non_seizure)
