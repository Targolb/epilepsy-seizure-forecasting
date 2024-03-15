import os
import numpy as np
import pyedflib
import glob

# Global variables
DATA_DIRECTORY = os.path.expanduser('/home/targol/EpilepticSeizur/physionet.org/files/chbmit/1.0.0')
SAVE_PATH = '/home/targol/EpilepticSeizur/data'
SEIZURE_FORECAST_WINDOW = 7200  # 2 hours in seconds


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
        current_file = None
        for i, line in enumerate(content):
            if line.startswith("File Name:"):
                current_file = line.split(": ")[1].strip()
                seizure_info[current_file] = []
            elif line.strip().startswith("Seizure Start Time"):
                start_time = int(line.split(": ")[1].strip().split()[0])  # Corrected
            elif line.strip().startswith("Seizure End Time"):
                end_time = int(line.split(": ")[1].strip().split()[0])  # Corrected
                seizure_info[current_file].append((start_time, end_time))

    return seizure_info


def extract_segments(data, fs, seizure_times, pre_seizure_window=7200, segment_length=300):
    pre_seizure_segments = []
    for start, _ in seizure_times:
        # Log total recording length for context
        total_recording_length_seconds = data.shape[1] / fs
        print(f"Total recording length: {total_recording_length_seconds} seconds")

        pre_seizure_start_time = max(0, start - pre_seizure_window)
        if pre_seizure_start_time == 0 and start < pre_seizure_window:
            print(f"Seizure at {start} seconds, too early for full {pre_seizure_window} seconds window.")
            # Implement fallback logic here if desired

        for segment_start in np.arange(pre_seizure_start_time, start, segment_length):
            segment_end = segment_start + segment_length
            if segment_end <= start:
                start_sample = int(segment_start * fs)
                end_sample = int(segment_end * fs)
                if end_sample <= data.shape[1]:
                    segment = data[:, start_sample:end_sample]
                    pre_seizure_segments.append(segment)
                else:
                    print(f"Segment from {segment_start} to {segment_end} seconds exceeds recording bounds.")

    if not pre_seizure_segments:
        print("No pre-seizure segments extracted.")
    else:
        print(f"Extracted {len(pre_seizure_segments)} pre-seizure segments.")

    return pre_seizure_segments


def save_segments(segments, base_path, file_name, segment_type):
    os.makedirs(base_path, exist_ok=True)
    for i, segment in enumerate(segments):
        segment_save_path = os.path.join(base_path, f"{file_name}_{segment_type}_{i}.npy")
        np.save(segment_save_path, segment)


def list_summary_files(data_directory):
    summary_files_pattern = os.path.join(data_directory, '*/chb*-summary.txt')
    return glob.glob(summary_files_pattern)


def preprocess_all():
    summary_files = list_summary_files(DATA_DIRECTORY)

    for summary_file in summary_files:
        print(f"Processing summary file: {summary_file}")
        seizure_info = parse_summary_file(summary_file)
        subject_id = os.path.basename(os.path.dirname(summary_file))

        for edf_file, seizures in seizure_info.items():
            full_path = os.path.join(DATA_DIRECTORY, subject_id, edf_file)
            if os.path.exists(full_path):
                print(f"Processing seizure file: {edf_file} for subject {subject_id}")
                data, labels, fs = load_eeg_file(full_path)
                pre_seizure_segments = extract_segments(data, fs, seizures, SEIZURE_FORECAST_WINDOW)
                if pre_seizure_segments:
                    save_segments(pre_seizure_segments, SAVE_PATH, edf_file, 'pre_seizure')
                else:
                    print("No pre-seizure segments extracted. Check extraction logic.")
            else:
                print(f"File {full_path} does not exist.")


if __name__ == "__main__":
    preprocess_all()
