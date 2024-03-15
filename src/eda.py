import os
import numpy as np
import matplotlib.pyplot as plt

data_directory = '/home/targol/EpilepticSeizur/data'
plot_directory = '/home/targol/EpilepticSeizur/data/plots'


def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists, create it if it doesn't.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_npy_files(data_dir):
    """
    Load .npy files from the specified directory.
    """
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    return file_paths


def plot_eeg_signal(data, title='EEG Signal', filename='plot.png'):
    """
    Plot and save EEG signal from a numpy array.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(data[0, :], label='EEG Channel 1')  # Assuming the first channel
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    # Save plot to file
    plt.savefig(os.path.join(plot_directory, filename))
    plt.close()


def perform_eda(npy_file_paths):
    """
    Perform exploratory data analysis on npy files.
    """
    ensure_directory_exists(plot_directory)
    for i, file_path in enumerate(npy_file_paths):
        data = np.load(file_path)
        print(f'Loaded {file_path}')
        print(f'Data Shape: {data.shape}')
        print(f'Mean: {np.mean(data, axis=1)}')
        print(f'Standard Deviation: {np.std(data, axis=1)}')

        # Generating filename for plot
        base_filename = os.path.basename(file_path).replace('.npy', '')
        plot_filename = f"{base_filename}_plot.png"

        # Plotting and saving the first EEG channel of the first segment
        plot_eeg_signal(data, title=f'EEG Signal for {os.path.basename(file_path)}', filename=plot_filename)


if __name__ == '__main__':
    npy_file_paths = load_npy_files(data_directory)
    print(f"Found {len(npy_file_paths)} .npy files for EDA.")
    perform_eda(npy_file_paths)
