import os
import numpy as np
import pyedflib
import glob

# Constants
SUBJECT_ID = "chb01"
DATA_DIRECTORY = os.path.expanduser(f'/home/targol/EpilepticSeizur/physionet.org/files/chbmit/1.0.0/{SUBJECT_ID}')
SAVE_PATH = f'/home/targol/EpilepticSeizur/data/{SUBJECT_ID}'
SEIZURE_FORECAST_WINDOW = 7200  # 1 hour in seconds


def load_eeg_file(filepath):
    with pyedflib.EdfReader(filepath) as f:
        n = f.signals_in_file
        sample_frequency = f.getSampleFrequency(0)
        data = np.zeros((n, f.getNSamples()[0]))
        for i in range(n):
            data[i, :] = f.readSignal(i)
    return data, f.getSignalLabels(), sample_frequency


def parse_summary_file(summary_file_path):
    seizure_info = {}
    with open(summary_file_path, 'r') as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if line.startswith("Seizure Start Time"):
                start_time = int(line.split(": ")[1].strip().split()[0])
            elif line.startswith("Seizure End Time"):
                end_time = int(line.split(": ")[1].strip().split()[0])
                # Assuming only one seizure info per summary file for simplicity
                seizure_info['start'] = start_time
                seizure_info['end'] = end_time
                break
    return seizure_info


def extract_and_save_segments(edf_filepath, seizure_info):
    if not seizure_info:
        print("No seizure information found.")
        return

    data, labels, fs = load_eeg_file(edf_filepath)
    pre_seizure_start_time = max(0, seizure_info['start'] - SEIZURE_FORECAST_WINDOW)

    segments = []
    segment_length = 300  # Segment length in seconds
    for segment_start in np.arange(pre_seizure_start_time, seizure_info['start'], segment_length):
        start_sample = int(segment_start * fs)
        end_sample = int(min(segment_start + segment_length, seizure_info['start']) * fs)
        segment = data[:, start_sample:end_sample]
        segments.append(segment)

    file_name = os.path.basename(edf_filepath).replace('.edf', '')
    save_segments(segments, SAVE_PATH, file_name, 'pre_seizure')
    print(f"Saved {len(segments)} segments for {file_name}")


def save_segments(segments, base_path, file_name, segment_type):
    os.makedirs(base_path, exist_ok=True)
    for i, segment in enumerate(segments):
        segment_save_path = os.path.join(base_path, f"{file_name}_{segment_type}_{i}.npy")
        np.save(segment_save_path, segment)


def preprocess_subject():
    summary_file = os.path.join(DATA_DIRECTORY, f'{SUBJECT_ID}-summary.txt')
    seizure_info = parse_summary_file(summary_file)

    edf_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.edf"))
    for edf_file in edf_files:
        extract_and_save_segments(edf_file, seizure_info)


if __name__ == "__main__":
    preprocess_subject()
