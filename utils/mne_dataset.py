"""
mne_dataset.py
-------------
MNE-based EEG dataset for loading EDF files with CSV annotations.
Generates CWT scalogram images for CNN training.

Supports:
- Binary mode: Normal vs Abnormal
- Three-class mode: Normal, Slowing Waves, Spike and Sharp waves

Usage:
    from utils.mne_dataset import prepare_dataset

    prepare_dataset(
        data_root="eeg_data/Data/NMT_Events",
        output_root="data",
        mode="three_class",
        window_duration=2.0,
        window_overlap=0.5,
        min_windows_per_subject=2,
        cwt_freqs=(1, 30, 30),
        img_size=(224, 224),
    )
"""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
try:
    import scipy.signal
except ImportError:
    scipy = None
try:
    import PIL.Image
    import PIL
except ImportError:
    PIL = None

logger = logging.getLogger(__name__)

LabelClasses3 = ["Normal", "Slowing Waves", "Spike and Sharp waves"]
LabelClasses2 = ["Normal", "Abnormal"]

LABEL_MAP_3CLASS = {
    "Normal": 0,
    "Slowing Waves": 1,
    "Spike and Sharp waves": 2,
}

LABEL_MAP_BINARY = {
    "Normal": 0,
    "Abnormal": 1,
}

CSV_LABEL_TO_CLASS = {
    "delta slow waves": "Slowing Waves",
    "sharp and slow wave": "Spike and Sharp waves",
    "sharp and slow waves": "Spike and Sharp waves",
    "spike and wave": "Spike and Sharp waves",
    "polyspikes and wave": "Spike and Sharp waves",
}


def parse_timestamp(ts: str) -> float:
    """Parse timestamp like '14:19:19:747' to seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) != 4:
        return 0.0

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(f"{parts[2]}.{parts[3]}")
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0


def load_csv_annotations(csv_path: Path) -> List[Dict]:
    """Parse CSV annotation file and return event windows."""
    events = []

    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return events

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_ts = row.get("Start time", "").strip()
            end_ts = row.get("End time", "").strip()
            label = row.get("Comment", "").strip().lower()

            if not start_ts or not end_ts:
                continue

            start_sec = parse_timestamp(start_ts)
            end_sec = parse_timestamp(end_ts)

            if end_sec <= start_sec:
                continue

            if label in CSV_LABEL_TO_CLASS:
                events.append(
                    {
                        "start": start_sec,
                        "end": end_sec,
                        "label": CSV_LABEL_TO_CLASS[label],
                    }
                )

    return events


def extract_windows(
    raw: mne.io.Raw,
    events: List[Dict],
    window_duration: float,
    window_overlap: float,
    sfreq: Optional[int] = None,
) -> List[Tuple[np.ndarray, str, float]]:
    """Extract windows from EDF based on event annotations."""
    if sfreq is None:
        sfreq = int(raw.info["sfreq"])

    total_duration = raw.times[-1]
    windows = []
    stride = window_duration * (1 - window_overlap)
    n_samples = int(window_duration * sfreq)

    for event in events:
        start_time = event["start"]
        end_time = event["end"]
        label = event["label"]

        current_time = start_time
        while current_time + window_duration <= end_time:
            start_sample = int(current_time * sfreq)
            end_sample = start_sample + n_samples

            if end_sample <= raw.n_times:
                data, times = raw[:, start_sample:end_sample]
                windows.append((data, label, current_time))

            current_time += stride

    if not events:
        for current_time in np.arange(0, total_duration - window_duration, stride):
            start_sample = int(current_time * sfreq)
            end_sample = start_sample + n_samples

            if end_sample <= raw.n_times:
                data, times = raw[:, start_sample:end_sample]
                windows.append((data, "Normal", current_time))

    return windows


def generate_cwt_scalogram(
    data: np.ndarray,
    sfreq: int,
    freq_range: Tuple[float, float] = (1, 30),
    n_freqs: int = 30,
    img_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Generate CWT scalogram from EEG data."""
    if scipy is None:
        raise ImportError("scipy is required for CWT generation")

    n_channels = data.shape[0]
    data = np.nan_to_num(data, nan=0.0)
    data = data - np.mean(data, axis=1, keepdims=True)
    data = data / (np.std(data, axis=1, keepdims=True) + 1e-8)

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

    scalograms = []
    for ch_data in data:
        f, t, Sxx = scipy.signal.spectrogram(
            ch_data,
            fs=sfreq,
            nperseg=min(256, len(ch_data) // 4),
            noverlap=0,
        )
        f_grid = np.linspace(0, sfreq / 2, len(f))
        Sxx_interp = np.zeros((n_freqs, Sxx.shape[1]))
        for j in range(Sxx.shape[1]):
            Sxx_interp[:, j] = np.interp(freqs, f_grid, Sxx[:, j])
        Sxx_log = np.log1p(np.abs(Sxx_interp))
        scalograms.append(Sxx_log)

    if not scalograms:
        scalogram = np.zeros((n_freqs, len(ch_data)))
    else:
        scalogram = np.mean(scalograms, axis=0)

    scalogram = (scalogram - scalogram.min()) / (
        scalogram.max() - scalogram.min() + 1e-8
    )

    fig, ax = plt.subplots(figsize=(img_size[0] / 96, img_size[1] / 96), dpi=96)
    ax.imshow(
        scalogram, aspect="auto", origin="lower", cmap="jet", interpolation="bilinear"
    )
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    image = image.reshape(height, width, 4)
    image = image[:, :, :3]
    plt.close(fig)

    from PIL import Image

    image = Image.fromarray(image).resize((img_size[1], img_size[0]), Image.BILINEAR)
    return np.array(image)


def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from EDF filename."""
    return re.sub(r"\.edf$", "", filename, flags=re.IGNORECASE)


def find_csv_for_edf(edf_path: Path, csv_dir: Path) -> Optional[Path]:
    """Find matching CSV for EDF file, handling different naming conventions."""
    subject_id = get_subject_id_from_filename(edf_path.name)

    # Try direct match
    direct = csv_dir / f"{subject_id}.csv"
    if direct.exists():
        return direct

    # Try without leading zeros
    id_num = re.sub(r"^0+", "", subject_id)
    no_zeros = csv_dir / f"{id_num}.csv"
    if no_zeros.exists():
        return no_zeros

    # Try extracting numeric part and matching
    num_match = re.search(r"\d+", subject_id)
    if num_match:
        num = num_match.group()
        for csv_file in csv_dir.glob("*.csv"):
            if csv_file.stem == num or csv_file.stem == num.lstrip("0"):
                return csv_file

    return None


def process_abnormal_subject(
    edf_path: Path,
    csv_path: Path,
    output_dir: Path,
    window_duration: float,
    window_overlap: float,
    cwt_freqs: Tuple[float, float, int],
    img_size: Tuple[int, int],
    min_windows: int,
    mode: str,
) -> int:
    """Process a single abnormal subject."""
    subject_id = get_subject_id_from_filename(edf_path.name)

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logger.warning(f"Failed to read {edf_path}: {e}")
        return 0

    raw.resample(100)

    events = load_csv_annotations(csv_path)
    if not events:
        return 0

    windows = extract_windows(raw, events, window_duration, window_overlap)
    if len(windows) < min_windows:
        return 0

    sfreq = int(raw.info["sfreq"])

    for idx, (data, label, start_time) in enumerate(windows):
        if mode == "binary":
            label_final = "Abnormal"
        else:
            label_final = label

        try:
            img = generate_cwt_scalogram(data, sfreq, cwt_freqs, cwt_freqs[2], img_size)
        except Exception as e:
            logger.warning(f"CWT failed for {subject_id} window {idx}: {e}")
            continue

        class_dir = output_dir / label_final
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{subject_id}_{idx}_{label_final.replace(' ', '_')}.png"
        filepath = class_dir / filename

        if filepath.exists():
            continue

        if PIL:
            PIL.Image.fromarray(img).save(filepath)
        else:
            raise ImportError("PIL is required for image saving")

    return len(windows)


def process_normal_subject(
    edf_path: Path,
    output_dir: Path,
    window_duration: float,
    window_overlap: float,
    total_duration: float,
    cwt_freqs: Tuple[float, float, int],
    img_size: Tuple[int, int],
    min_windows: int,
    mode: str,
) -> int:
    """Process a single normal subject."""
    subject_id = get_subject_id_from_filename(edf_path.name)

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        logger.warning(f"Failed to read {edf_path}: {e}")
        return 0

    raw.resample(100)

    sfreq = int(raw.info["sfreq"])
    n_samples = int(window_duration * sfreq)
    stride = window_duration * (1 - window_overlap)

    windows_count = 0
    for current_time in np.arange(0, total_duration - window_duration, stride):
        start_sample = int(current_time * sfreq)
        end_sample = start_sample + n_samples

        if end_sample > raw.n_times:
            break

        data, _ = raw[:, start_sample:end_sample]

        try:
            img = generate_cwt_scalogram(data, sfreq, cwt_freqs, cwt_freqs[2], img_size)
        except Exception as e:
            logger.warning(f"CWT failed for {subject_id}: {e}")
            continue

        class_dir = output_dir / "Normal"
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{subject_id}_{windows_count}_Normal.png"
        filepath = class_dir / filename

        if filepath.exists():
            windows_count += 1
            continue

        if PIL:
            PIL.Image.fromarray(img).save(filepath)
        else:
            raise ImportError("PIL is required for image saving")
        windows_count += 1

    if windows_count < min_windows:
        for f in (output_dir / "Normal").glob(f"{subject_id}_*_Normal.png"):
            f.unlink()
        return 0

    return windows_count


def split_subjects(
    abnormal_files: List[Path],
    normal_files: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split subjects by filename for subject-wise train/val/test split.

    Each group (abnormal/normal) is split independently:
    - 70% train, 20% val, 10% test for abnormal
    - 70% train, 20% val, 10% test for normal
    """
    abnormal_sorted = sorted(abnormal_files, key=lambda x: x.name)
    normal_sorted = sorted(normal_files, key=lambda x: x.name)

    n_abnormal = len(abnormal_sorted)
    n_normal = len(normal_sorted)

    n_abnormal_train = int(n_abnormal * train_ratio)
    n_abnormal_val = int(n_abnormal * val_ratio)

    n_normal_train = int(n_normal * train_ratio)
    n_normal_val = int(n_normal * val_ratio)

    train_ab = abnormal_sorted[:n_abnormal_train]
    val_ab = abnormal_sorted[n_abnormal_train : n_abnormal_train + n_abnormal_val]
    test_ab = abnormal_sorted[n_abnormal_train + n_abnormal_val :]

    train_norm = normal_sorted[:n_normal_train]
    val_norm = normal_sorted[n_normal_train : n_normal_train + n_normal_val]
    test_norm = normal_sorted[n_normal_train + n_normal_val :]

    return (train_ab, val_ab, test_ab), (train_norm, val_norm, test_norm)


def prepare_dataset(
    data_root: str = "eeg_data/Data/NMT_Events",
    output_root: str = "data",
    mode: str = "three_class",
    window_duration: float = 2.0,
    window_overlap: float = 0.5,
    min_windows_per_subject: int = 2,
    cwt_freqs: Tuple[float, float, int] = (1, 30, 30),
    img_size: Tuple[int, int] = (224, 224),
    total_duration: float = 600.0,
    train_ids: Optional[List[int]] = None,
    valid_ids: Optional[List[int]] = None,
    test_ids: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Prepare the dataset with explicit train/val/test split."""
    if train_ids is None or valid_ids is None or test_ids is None:
        raise ValueError("train_ids, valid_ids, and test_ids are required")

    all_ids = set(train_ids) | set(valid_ids) | set(test_ids)
    if len(all_ids) != len(train_ids) + len(valid_ids) + len(test_ids):
        raise ValueError("train_ids, valid_ids, and test_ids must not overlap")

    data_root = Path(data_root)
    output_root = Path(output_root)

    edf_abnormal_dir = data_root / "edf" / "Abnormal EDF Files"
    edf_normal_dir = data_root / "edf" / "Normal EDF Files"
    csv_dir = data_root / "csv" / "SW & SSW CSV Files"

    abnormal_files = sorted(edf_abnormal_dir.glob("*.edf"))
    normal_files = sorted(edf_normal_dir.glob("*.edf"))

    all_ids_str = {f"{i:07d}" for i in all_ids}
    abnormal_files = [f for f in abnormal_files if f.stem in all_ids_str]
    normal_files = [f for f in normal_files if f.stem in all_ids_str]

    valid_abnormal = []
    for edf_path in abnormal_files:
        csv_path = find_csv_for_edf(edf_path, csv_dir)
        if csv_path and csv_path.exists():
            valid_abnormal.append(edf_path)
    abnormal_files = valid_abnormal

    train_ids_str = {f"{i:07d}" for i in train_ids}
    valid_ids_str = {f"{i:07d}" for i in valid_ids}
    test_ids_str = {f"{i:07d}" for i in test_ids}

    train_ab = [f for f in abnormal_files if f.stem in train_ids_str]
    val_ab = [f for f in abnormal_files if f.stem in valid_ids_str]
    test_ab = [f for f in abnormal_files if f.stem in test_ids_str]

    train_norm = [f for f in normal_files if f.stem in train_ids_str]
    val_norm = [f for f in normal_files if f.stem in valid_ids_str]
    test_norm = [f for f in normal_files if f.stem in test_ids_str]

    logger.info(
        f"Found {len(abnormal_files)} abnormal and {len(normal_files)} normal EDF files"
    )

    logger.info(
        f"Split: Train={len(train_ab) + len(train_norm)}, "
        f"Val={len(val_ab) + len(val_norm)}, Test={len(test_ab) + len(test_norm)}"
    )

    splits = [
        ("train", train_ab, train_norm),
        ("valid", val_ab, val_norm),
        ("test", test_ab, test_norm),
    ]

    stats = {}

    for split_name, abnormal_subjects, normal_subjects in splits:
        logger.info(f"Processing split: {split_name}")
        output_dir = output_root / split_name

        for edf_path in tqdm(abnormal_subjects, desc=f"Abnormal-{split_name}"):
            csv_path = find_csv_for_edf(edf_path, csv_dir)
            n_windows = process_abnormal_subject(
                edf_path,
                csv_path,
                output_dir,
                window_duration,
                window_overlap,
                cwt_freqs,
                img_size,
                min_windows_per_subject,
                mode,
            )

        for edf_path in tqdm(normal_subjects, desc=f"Normal-{split_name}"):
            n_windows = process_normal_subject(
                edf_path,
                output_dir,
                window_duration,
                window_overlap,
                total_duration,
                cwt_freqs,
                img_size,
                min_windows_per_subject,
                mode,
            )

        for label_name in LabelClasses3 if mode == "three_class" else LabelClasses2:
            n_files = len(list(output_dir.glob(f"{label_name}/*.png")))
            stats[f"{split_name}_{label_name}"] = n_files

    logger.info(f"Dataset prepared: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prepare_dataset(
        data_root="eeg_data/Data/NMT_Events",
        output_root="data",
        mode="three_class",
    )
