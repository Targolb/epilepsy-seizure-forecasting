import os
import numpy as np
import pyedflib
import glob
import random

# Constants
SUBJECT_ID = "chb01"
DATA_DIRECTORY = os.path.expanduser(f'/home/targol/EpilepticSeizur/physionet.org/files/chbmit/1.0.0/{SUBJECT_ID}')
SAVE_PATH = f'/home/targol/EpilepticSeizur/data/{SUBJECT_ID}'
SEIZURE_FORECAST_WINDOW = 3600  # 1 hour in seconds
SEGMENT_LENGTH = 300  # Segment length in seconds


def load_eeg_file(filepath):
    with pyedflib.EdfReader(filepath) as f:
        n = f.signals_in_file
        sample_frequency = f.getSampleFrequency(0)
        signal_labels = f.getSignalLabels()
        data = np.zeros((n, f.getNSamples()[0]))
        for i in range(n):
            data[i, :] = f.readSignal(i)
    return data, signal_labels, sample_frequency


def parse_summary_file(summary_file_path):
    seizures = []
    with open(summary_file_path, 'r') as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if line.startswith("Seizure Start Time"):
                start_time = int(line.split(": ")[1].strip().split()[0])
            elif line.startswith("Seizure End Time"):
                end_time = int(line.split(": ")[1].strip().split()[0])
                seizures.append((start_time, end_time))
    return seizures


def extract_segments(data, fs, seizure_times):
    pre_seizure_segments = []
    non_seizure_segments = []

    for start, end in seizure_times:
        pre_seizure_start_time = max(0, start - SEIZURE_FORECAST_WINDOW)
        for segment_start in np.arange(pre_seizure_start_time, start, SEGMENT_LENGTH):
            start_sample = int(segment_start * fs)
            end_sample = int(min(segment_start + SEGMENT_LENGTH, start) * fs)
            segment = data[:, start_sample:end_sample]
            pre_seizure_segments.append(segment)

    # Assuming non-seizure segments should equal the number of pre-seizure segments
    total_recording_length = data.shape[1] / fs
    available_for_non_seizure = total_recording_length - sum([end - start for start, end in seizure_times])
    max_non_seizure_segments = int(available_for_non_seizure / SEGMENT_LENGTH)

    non_seizure_needed = min(len(pre_seizure_segments), max_non_seizure_segments)
    while len(non_seizure_segments) < non_seizure_needed:
        random_start = random.randint(0, total_recording_length - SEGMENT_LENGTH)
        if all(not (start <= random_start + SEGMENT_LENGTH and end >= random_start) for start, end in seizure_times):
            start_sample = int(random_start * fs)
            end_sample = start_sample + int(SEGMENT_LENGTH * fs)
            segment = data[:, start_sample:end_sample]
            non_seizure_segments.append(segment)

    return pre_seizure_segments, non_seizure_segments


def save_segments(segments, base_path, file_name, segment_type):
    os.makedirs(base_path, exist_ok=True)
    for i, segment in enumerate(segments):
        segment_save_path = os.path.join(base_path, f"{file_name}_{segment_type}_{i}.npy")
        np.save(segment_save_path, segment)


def preprocess_subject():
    summary_file = os.path.join(DATA_DIRECTORY, f'{SUBJECT_ID}-summary.txt')
    seizure_times = parse_summary_file(summary_file)

    edf_files = glob.glob(os.path.join(DATA_DIRECTORY, "*.edf"))
    for edf_file in edf_files:
        data, labels, fs = load_eeg_file(edf_file)
        pre_seizure_segments, non_seizure_segments = extract_segments(data, fs, seizure_times)

        file_name = os.path.basename(edf_file).replace('.edf', '')
        save_segments(pre_seizure_segments, SAVE_PATH, file_name, 'pre_seizure')
        save_segments(non_seizure_segments, SAVE_PATH, file_name, 'non_seizure')
        print(
            f"Saved {len(pre_seizure_segments)} pre-seizure and {len(non_seizure_segments)} non-seizure segments for {file_name}")


if __name__ == "__main__":
    preprocess_subject()
