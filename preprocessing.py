import os
import re
import gc
import mne
import wfdb
import tqdm
import numpy as np
import random
from scipy import signal
import h5py
import logging
from typing import Dict, List
import matplotlib.pyplot as plt

# === Settings ===
ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
             'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
             'FZ-CZ', 'CZ-PZ']
time_window = 1.0  # seconds
time_step = 0.5  # seconds
preictal_duration = 30 * 60  # 30 minutes
data_dir = ''#path of dataset 
nperseg = 18  # Adjusted for ~10 frequency bins (nperseg//2 + 1 = 10)
noverlap = 9
min_class_ratio = 0.5  # Lowered to include more windows
target_samples_per_class = 25000  

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Helper Functions ===
def validate_channels(raw: mne.io.Raw, ch_labels: List[str]) -> bool:
    missing = []
    for ch in ch_labels:
        matches = [l for l in raw.ch_names if re.search(ch, l, re.IGNORECASE)]
        if not matches:
            missing.append(ch)
    if missing:
        logging.warning(f"Missing channels: {missing}")
        return False
    return True

def process_file(file_path: str) -> Dict[int, List[np.ndarray]]:
    """Process a single EDF file and return spectrograms by class."""
    spectrograms_by_class = {0: [], 1: [], 2: []}
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        if not validate_channels(raw, ch_labels):
            logging.warning(f"Skipping file {file_path} due to missing channels")
            return spectrograms_by_class

        fs = int(raw.info['sfreq'])
        win_len = int(time_window * fs)
        step = int(time_step * fs)
        total_len = raw.n_times

        # Channel mapping
        ch_mapping = {}
        for ch in ch_labels:
            matches = [l for l in raw.ch_names if re.search(ch, l, re.IGNORECASE)]
            if matches:
                ch_mapping[matches[0]] = ch
        raw.rename_channels(ch_mapping)
        raw.pick_channels(ch_labels, ordered=True)
        data = raw.get_data(picks=ch_labels) * 1e6  # Convert to microvolts

        # Create label array
        labels_array = np.zeros(total_len, dtype=np.int8)
        if os.path.exists(file_path + '.seizures'):
            try:
                ann = wfdb.rdann(file_path, 'seizures')
                for i in range(len(ann.sample) // 2):
                    sz_start = ann.sample[i * 2]
                    sz_end = ann.sample[i * 2 + 1]
                    preictal_start = max(0, sz_start - int(preictal_duration * fs))
                    labels_array[preictal_start:sz_start] = 1
                    labels_array[sz_start:sz_end] = 2
            except Exception as e:
                logging.warning(f"Error reading seizures for {file_path}: {str(e)}")

        # Per-window labels
        labels_per_window = []
        for i in range(0, total_len - win_len, step):
            window_labels = labels_array[i:i + win_len]
            if len(window_labels) == win_len:
                counts = np.bincount(window_labels, minlength=3)
                if counts.max() / win_len >= min_class_ratio:
                    labels_per_window.append(counts.argmax())
                else:
                    labels_per_window.append(-1)
            else:
                labels_per_window.append(-1)
        labels_per_window = np.array(labels_per_window)

        # Extract segments
        for i, label in enumerate(labels_per_window):
            if label == -1:
                continue
            segment = data[:, i * step:i * step + win_len]
            if segment.shape[1] != win_len:
                continue
            specs = []
            for ch in range(segment.shape[0]):
                f, t, Zxx = signal.stft(segment[ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
                specs.append(np.abs(Zxx).astype(np.float32))
            spec_tensor = np.stack(specs, axis=-1)  # (freq, time, channels)
            spectrograms_by_class[label].append(spec_tensor)

        logging.info(f"File {file_path}: {len(spectrograms_by_class[0])} interictal, "
                     f"{len(spectrograms_by_class[1])} preictal, "
                     f"{len(spectrograms_by_class[2])} ictal samples")
        raw.close()
        gc.collect()
        return spectrograms_by_class
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return spectrograms_by_class

def augment_data(spectrograms: List[np.ndarray], target_size: int, class_name: str) -> List[np.ndarray]:
    """Augment spectrograms to reach target size."""
    if not spectrograms:
        logging.warning(f"No samples for {class_name}. Returning empty list.")
        return []
    augmented = []
    while len(augmented) + len(spectrograms) < target_size:
        s = random.choice(spectrograms)
        # Amplitude scaling
        augmented.append(s * random.uniform(0.8, 1.2))
        # Frequency jitter
        augmented.append(np.roll(s, shift=random.randint(-2, 2), axis=0))
    logging.info(f"Augmented {class_name} from {len(spectrograms)} to {target_size} samples")
    return spectrograms + augmented[:target_size - len(spectrograms)]

# === Load and process files ===
files_train = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.edf')]
files_train.sort()

spectrograms_by_class = {0: [], 1: [], 2: []}
for file_path in tqdm.tqdm(files_train, desc="Processing EDF files"):
    result = process_file(file_path)
    for cls in [0, 1, 2]:
        spectrograms_by_class[cls].extend(result[cls])

# Log class distribution
for cls in [0, 1, 2]:
    logging.info(f"Class {cls} ({['interictal', 'preictal', 'ictal'][cls]}): "
                 f"{len(spectrograms_by_class[cls])} samples")

# === Balance classes ===
logging.info(f"Target samples per class: {target_samples_per_class}")

spectrograms = []
labels = []
class_names = ['interictal', 'preictal', 'ictal']
for cls in [0, 1, 2]:
    samples = spectrograms_by_class[cls]
    class_name = class_names[cls]
    if not samples:
        logging.warning(f"No samples for {class_name}. Generating {target_samples_per_class} augmented samples.")
        # Use preictal samples as a fallback for ictal if empty
        if cls == 2 and spectrograms_by_class[1]:
            samples = spectrograms_by_class[1][:min(len(spectrograms_by_class[1]), target_samples_per_class)]
            samples = [s * random.uniform(1.2, 1.5) for s in samples]  # Simulate ictal amplitude increase
        elif cls == 1 and spectrograms_by_class[0]:
            samples = spectrograms_by_class[0][:min(len(spectrograms_by_class[0]), target_samples_per_class)]
            samples = [s * random.uniform(1.0, 1.2) for s in samples]  # Simulate preictal
    if len(samples) < target_samples_per_class:
        samples = augment_data(samples, target_samples_per_class, class_name)
    elif len(samples) > target_samples_per_class:
        samples = random.sample(samples, target_samples_per_class)
        logging.info(f"Subsampled {class_name} from {len(spectrograms_by_class[cls])} to {target_samples_per_class}")
    spectrograms.extend(samples)
    labels.extend([class_name] * len(samples))

if not spectrograms:
    logging.error("No samples generated for any class. Check dataset or preprocessing steps.")
    raise ValueError("No samples available after balancing.")

spectrograms = np.array(spectrograms, dtype=np.float32)
labels = np.array(labels, dtype='S10')  
# === Save to HDF5 ===
logging.info(f"Saving {spectrograms.shape[0]} samples to HDF5...")
with h5py.File("spectrograms_labels_balanced.h5", "w") as hf:
    hf.create_dataset("spectrograms", data=spectrograms, compression="gzip")
    hf.create_dataset("labels", data=labels, compression="gzip")
    hf.attrs["sampling_rate"] = 256
    hf.attrs["channels"] = ch_labels
    hf.attrs["time_window"] = time_window
    hf.attrs["nperseg"] = nperseg

logging.info("✅ Done saving.")
logging.info(f"Shape: {spectrograms.shape}")
# Decode byte strings for class balance logging
decoded_labels = [l.decode('utf-8') if isinstance(l, bytes) else l for l in labels]
logging.info(f"Class balance: {np.bincount(np.array([class_names.index(l) for l in decoded_labels]))}")

# === Visualization ===
samples_to_plot = [0, 1, 2]
class_names_display = ["Interictal", "Preictal", "Ictal"]
channel_idx = 0

for cls in samples_to_plot:
    # Decode labels for comparison
    label_indices = [class_names.index(l.decode('utf-8') if isinstance(l, bytes) else l) for l in labels]
    if cls not in label_indices:
        logging.warning(f"No samples for {class_names_display[cls]}. Skipping visualization.")
        continue
    idx = np.where(np.array(label_indices) == cls)[0][0]
    spec = spectrograms[idx]
    raw_data = np.mean(spec[:, :, channel_idx], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    time = np.linspace(0, time_window, spec.shape[1])
    axes[0].plot(time, raw_data)
    axes[0].set_title(f"{class_names_display[cls]} - Channel: {ch_labels[channel_idx]}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (μV)")

    freqs = np.linspace(0, 256 / 2, spec.shape[0])
    times = np.linspace(0, time_window, spec.shape[1])
    axes[1].imshow(spec[:, :, channel_idx], aspect='auto', origin='lower',
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='viridis')
    axes[1].set_title(f"Spectrogram - {class_names_display[cls]}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")

    plt.suptitle(f"Sample {cls} - {class_names_display[cls]}", fontsize=14)
    plt.show()