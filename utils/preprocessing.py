"""
preprocessing.py
---------------
Window extraction and CWT scalogram generation for EEG anomaly classification.

EDF files are split into 50% overlapping windows of 400 samples (2 seconds at 200 Hz).
Labels from CSV annotations are mapped to super-classes and assigned per window.
CWT scalograms are computed using multiple wavelet types and saved as PNG images.

Classes
-------
- WindowExtractor  : Extract windows and labels from EDF/CSV pairs.
- CWTProcessor     : Compute CWT scalograms and save as images.

Functions
---------
- extract_labels_from_csv   : Parse CSV annotation file.
- clean_labels              : Map raw labels to super-classes.
- generate_label_array      : Create per-channel label arrays.
- create_overlapping_epochs : Split data into overlapping windows.
- threshold_epoch_labels    : Assign one label per epoch via threshold.
- encode_labels            : Convert string labels to integers.

Constants
---------
- SAMPLING_RATE  : 200 Hz
- WINDOW_SIZE    : 400 samples (2 seconds)
- CHANNELS       : 19 EEG channel names
- LABEL_CLASSES  : Super-class label names
- WAVELET_TYPES  : CWT wavelet names
- SCALES         : CWT scale range
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLING_RATE = 200
WINDOW_SIZE = 400
OVERLAP = 0.5
CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "PZ",
    "CZ",
]
LABEL_CLASSES = ["Normal", "Delta Slow Wave", "Sharp Wave", "Spike and Wave Discharge"]
DEST_LIST = ["Normal", "Slowing Waves", "Spike and Sharp waves"]
DEST_LIST_BINARY = ["Normal", "Abnormal"]
WAVELET_TYPES = ["mexh", "morl", "gaus1", "gaus2"]
SCALES = np.arange(1, 24)


def extract_labels_from_csv(
    name: Union[int, str],
    csvdir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a CSV annotation file and return label types, channels, and timestamps.

    Args:
        name      : EDF file ID (int or str) used to locate the CSV file.
        csvdir    : Directory containing CSV annotation files.

    Returns:
        type      : Array of label types (e.g., 'slow waves', 'spike').
        channels  : Array of channel names for each annotation.
        beg       : Array of start sample indices.
        end       : Array of end sample indices.
    """
    if not isinstance(name, str):
        name = str(name) + ".csv"
    if csvdir[-1] != "/":
        csvdir = csvdir + "/"

    csv = pd.read_csv(csvdir + name)
    offset = csv["File Start"][0].split(":")

    beg = csv["Start time"].str.split(":").to_numpy()
    end = csv["End time"].str.split(":").to_numpy()
    channels = csv["Channel names"].str.split().to_numpy()
    label_types = csv["Comment"].to_numpy()

    offset = [int(o) for o in offset]

    for i in range(beg.shape[0]):
        if len(beg[i]) < 3:
            continue
        for j in range(len(beg[i])):
            beg[i][j] = int(beg[i][j])
            if len(beg[i]) < 4:
                beg[i].append(0)
        for j in range(len(end[i])):
            end[i][j] = int(end[i][j])
            if len(end[i]) < 4:
                end[i].append(0)
        for j in range(len(offset)):
            beg[i][j] -= offset[j]
            end[i][j] -= offset[j]
        beg[i] = int(
            (beg[i][0] * 3600 + beg[i][1] * 60 + beg[i][2] + beg[i][3] / 1000)
            * SAMPLING_RATE
        )
        end[i] = int(
            (end[i][0] * 3600 + end[i][1] * 60 + end[i][2] + end[i][3] / 1000)
            * SAMPLING_RATE
        )

    return label_types, channels, np.array(beg), np.array(end)


def clean_labels(labels: np.ndarray) -> List[str]:
    """Map raw annotation labels to predefined super-classes.

    Categories:
        - Normal
        - Delta Slow Wave (slow waves, delta waves, etc.)
        - Sharp Wave (sharp waves, generalized sharp waves)
        - Spike and Wave Discharge (spike/wave patterns)

    Args:
        labels : Array of raw label strings.

    Returns:
        List of cleaned label strings.
    """
    label_dict = [
        ["No Comment", "delete previous", "nan", "Unknown", "Normal"],
        [
            "sharp waves",
            "sharp wave",
            "generalized sharp waves",
            "generalized sharp waves discharge",
            "Sharp Wave",
        ],
        [
            "delta slow wave",
            "delta waves",
            "delta slow waves",
            "sharp and delta slow waves",
            "sharp and delta waves",
            "sharp and delta wave",
            "sharp and slow waves",
            "sharp and slow wave",
            "generalized paroxymal delta slow waves",
            "generalized paroxysmal delta slow waves",
            "generalized parosysmal delta slow waves",
            "generalized delta slow waves",
            "slow waves",
            "generalized delta slow waves ",
            "paroxysmal delta slow waves",
            " delta slow waves",
            "paroxysmal generalized delta slow waves",
            "paroxysmal generalized deta slow waves",
            "Delta ",
            "Delta Slow Wave",
        ],
        [
            "2 hertz slow spike and wave discharge",
            "spike wave",
            "spikes",
            "polypspikes and wave",
            "polyspike and wave",
            "polyspikes and wave",
            "generalized paroxysmal spike and wave discharge",
            "generalized paroxymal spike and wave discharge",
            "fragmented spike and wave discharge",
            "generalized paroxysmal 3 hertz spike and wave discharge",
            "generalized paroxysmal  spike and wave discharge",
            "generalized spike and wave discharge",
            "generalized spike and wave discharges",
            "generalized 4 hertz spike and wave discharge",
            "spike and wave",
            "spike and wave discharge",
            "Generalized 3 hertz spike and wave",
            "generalized 3 hertz spike and wave discharge",
            "generalized 3 hertz spike and wave activity",
            "generalized 2 hertz spike and wave discharge",
            "generalized 2 hertz spike and wave",
            "2 hertz spike and wave discharge",
            "spike and waves",
            "3 hertz fragmented spike and wave discharge",
            "generalized spike and wave ",
            "generalized spike and wave",
            "generalized spike and wave activity",
            "spike and wave ",
            "polyspikes discharge",
            "Generalized  paroxysmal 4 hertz spike and wave discharge",
            "generalized 3.5 hertz spike and wave discharge",
            "generalized 3 hertz spike and wave discharges",
            "generalized 2 hertz spike and wave discharges",
            "3 hertz spike and wave discharge",
            " 3 hertz spike and wave discharge",
            "paroxysmal generalized 3.5 spike and wave discharge",
            "paroxysmal generalized 3.5 hertz spike and wave discharge",
            " spike and wave discharge",
            "generalized spike  and wave discharge",
            "generalized  3 hertz spike and wave discharge",
            "generalized  spike and wave discharge",
            "generalized 3 hertz  spike and wave discharge",
            "spike an dwave",
            "spike",
            "Paroxysmal generalized 3 hertz spike and wave discharge",
            "paroxysmal generalized spike and wave discharge",
            "polyspikes",
            "generalized polyspike discharge",
            "generalized polyspikes discharge",
            "generalized 4 hertz spike and wave discharge",
            "rolandic spike",
            "rolandic spikes",
            "generalized 3 hertz spike and wave",
            "polyspikes and wave",
            "spike wave",
            "polyspikes ",
            "Spike and Wave Discharge",
        ],
        ["Beta waves", "beta waves", "Beta Wave"],
        ["theta waves", "Theta Wave"],
        ["triphasic waves", "Triphasic Wave"],
        ["burst suppression", "Burst Suppression"],
        ["low voltage", "no waveform", "Low Voltage"],
    ]

    ret = []
    for label in labels:
        matched = False
        for group in label_dict:
            for pattern in group:
                if pattern.lower().strip() == str(label).lower().strip():
                    ret.append(group[-1])
                    matched = True
                    break
            if matched:
                break
        if not matched:
            print(f"Not found: {label}")

    if len(ret) != len(labels):
        print(labels)
        print("Did not catch all labels. Please check.")
    return ret


def generate_label_array(labels: List[str], channels: np.ndarray) -> List[List[str]]:
    """Create per-channel label arrays matching EEG electrode positions.

    Args:
        labels   : Cleaned label strings for each annotation event.
        channels : Channel names from CSV for each annotation event.

    Returns:
        List of per-channel label lists for each annotation event.
    """
    channel_list = CHANNELS
    ret = []
    for i in range(len(labels)):
        ret.append([])
        for ch in channel_list:
            if ch in channels[i]:
                ret[-1].append(labels[i])
            else:
                ret[-1].append("Normal")
    return ret


def create_overlapping_epochs(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> Tuple[List[np.ndarray], List[int]]:
    """Split EEG data into overlapping windows (epochs).

    Args:
        data       : EEG data array of shape (n_channels, n_samples).
        window_size: Number of samples per window (default: 400).
        overlap    : Fraction of overlap between adjacent windows (default: 0.5).

    Returns:
        windows    : List of epoch arrays, each shape (n_channels, window_size).
        indices    : List of start sample indices for each window.
    """
    n_channels, n_samples = data.shape
    step = int(window_size * (1 - overlap))
    windows = []
    indices = []

    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[:, start:end])
        indices.append(start)

    return windows, indices


def threshold_epoch_labels(
    epoch_labels: np.ndarray,
    threshold_pct: float = 25.0,
) -> np.ndarray:
    """Assign a single label to each epoch based on label abundance.

    An epoch is labeled as the most frequent non-zero label if at least
    `threshold_pct` of its samples contain that label.

    Args:
        epoch_labels : Array of shape (n_epochs, n_channels, window_size).
                       Values are label indices (0 = Normal, 1, 2 = anomaly types).
        threshold_pct: Percentage threshold for labeling (default: 25.0).

    Returns:
        Array of shape (n_epochs, n_channels) with integer labels per epoch.
    """
    n_epochs, n_channels, window_size = epoch_labels.shape
    threshold = round(window_size * (threshold_pct / 100))
    label_array = []

    for ch in range(n_channels):
        epoch_label_list = []
        for ep in range(n_epochs):
            non_zero_count = np.count_nonzero(epoch_labels[ep, ch])
            if non_zero_count > threshold:
                ab_type = np.unique(epoch_labels[ep, ch])[-1]
                epoch_label_list.append(int(ab_type))
            else:
                epoch_label_list.append(0)
        label_array.append(epoch_label_list)

    label_array = np.array(label_array, dtype=int)
    return label_array.T


def encode_labels(label_data: np.ndarray) -> np.ndarray:
    """Convert string label arrays to integer encoding.

    Mapping:
        'Normal'              -> 0
        'Delta Slow Wave'     -> 1 (Slowing Waves)
        'Sharp Wave'          -> 2 (Spike and Sharp Waves)
        'Spike and Wave Discharge' -> 2 (Spike and Sharp Waves)

    Args:
        label_data : Array of string labels.

    Returns:
        Integer-encoded label array.
    """
    encoded = label_data.copy()
    encoded[encoded == "Normal"] = 0
    encoded[encoded == "Delta Slow Wave"] = 1
    encoded[encoded == "Sharp Wave"] = 2
    encoded[encoded == "Spike and Wave Discharge"] = 2
    return encoded.astype(int)


def encode_labels_binary(label_data: np.ndarray) -> np.ndarray:
    """Convert 3-class encoded labels to 2-class by merging abnormal labels.

    Mapping:
        0 (Normal)             -> 0 (Normal)
        1 (Slowing Waves)     -> 1 (Abnormal)
        2 (Spike/Sharp Waves) -> 1 (Abnormal)

    Args:
        label_data : 3-class encoded array (values: 0, 1, 2).

    Returns:
        2-class encoded label array (values: 0, 1).
    """
    encoded = encode_labels(label_data).copy()
    encoded[encoded == 2] = 1
    return encoded


class WindowExtractor:
    """Extract overlapping EEG windows with per-channel labels from EDF/CSV pairs.

    Parameters
    ----------
    sampling_rate : int
        EEG sampling frequency in Hz (default: 200).
    window_size   : int
        Number of samples per window (default: 400 = 2 seconds).
    overlap       : float
        Fraction of overlap between windows (default: 0.5 = 50%).
    min_ab_threshold : float
        Minimum fraction of abnormality duration to affect a window
        during label broadcasting (default: 0.7).
    epoch_threshold : float
        Percentage threshold for assigning an epoch its majority label
        (default: 25.0).

    Attributes
    ----------
    epochs   : List[np.ndarray]
        Extracted windows, each (n_channels, window_size).
    labels   : np.ndarray
        Encoded labels of shape (n_epochs, n_channels).
    indices  : List[int]
        Start sample index for each epoch.

    Example
    -------
    >>> extractor = WindowExtractor()
    >>> extractor.process(edf_path="data/0001.edf", csv_path="annotations/")
    >>> print(extractor.epochs.shape, extractor.labels.shape)
    """

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
        min_ab_threshold: float = 0.7,
        epoch_threshold: float = 25.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.min_ab_threshold = min_ab_threshold
        self.epoch_threshold = epoch_threshold

        self.epochs: List[np.ndarray] = []
        self.labels: Optional[np.ndarray] = None
        self.indices: List[int] = []
        self._epoch_label_array: List[List[List[int]]] = []

    def process(
        self,
        edf_path: Union[str, Path],
        csv_path: Union[str, Path],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load EDF file and CSV annotations, extract windows and labels.

        Args:
            edf_path : Path to EDF file.
            csv_path : Directory containing CSV annotation files.

        Returns:
            epochs : List of epoch arrays.
            labels : Encoded labels (n_epochs, n_channels).
        """
        edf_path = Path(edf_path)
        csv_path = Path(csv_path)

        import mne

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        data, _ = raw[: len(CHANNELS), :]
        data = data.astype(np.float64)

        csv_name = edf_path.stem + ".csv"
        types, ch_annots, beg, end = extract_labels_from_csv(csv_name, str(csv_path))
        cleaned = clean_labels(types)
        channel_labels = generate_label_array(cleaned, ch_annots)
        channel_labels = np.array(channel_labels)

        n_channels, n_samples = data.shape
        nsamples = self.window_size

        window_size = self.window_size
        overlap = self.overlap
        step = int(window_size * (1 - overlap))

        self.epochs = []
        self._epoch_label_array = []
        self.indices = []

        n_windows = (n_samples - window_size) // step + 1

        for i in range(n_windows):
            start = i * step
            end_idx = start + window_size
            self.epochs.append(data[:, start:end_idx])
            self._epoch_label_array.append(
                [["Normal"] * n_channels for _ in range(n_windows)]
            )
            self.indices.append(start)

        new_end = np.delete(end, np.argwhere(end / nsamples >= n_windows - 1))
        new_beg = beg[0 : len(new_end)]

        for j in range(len(new_beg)):
            annot_label = channel_labels[j]

            window_index_beg = new_beg[j] / nsamples
            if (1 - window_index_beg % 1) > self.min_ab_threshold:
                window_index_beg = int(window_index_beg)
            else:
                window_index_beg = math.ceil(window_index_beg)

            window_index_end = new_end[j] / nsamples
            if window_index_end % 1 > self.min_ab_threshold:
                window_index_end = math.ceil(window_index_end)
            else:
                window_index_end = int(window_index_end)

            for ch_idx in range(n_channels):
                label = annot_label[ch_idx]
                if label == "Normal":
                    continue
                for win_idx in range(window_index_beg, window_index_end + 1):
                    if 0 <= win_idx < n_windows:
                        self._epoch_label_array[win_idx][ch_idx] = label

        self.labels = self._build_epoch_labels()
        return self.epochs, self.labels

    def _build_epoch_labels(self) -> np.ndarray:
        n_epochs = len(self.epochs)
        n_channels = self.epochs[0].shape[0]

        epoch_labels = np.zeros((n_epochs, n_channels, self.window_size), dtype=int)
        for i, win_labels in enumerate(self._epoch_label_array):
            for ch, label in enumerate(win_labels):
                if label == "Normal":
                    epoch_labels[i, ch, :] = 0
                elif label == "Delta Slow Wave":
                    epoch_labels[i, ch, :] = 1
                elif label == "Sharp Wave":
                    epoch_labels[i, ch, :] = 2
                elif label == "Spike and Wave Discharge":
                    epoch_labels[i, ch, :] = 2

        return threshold_epoch_labels(epoch_labels, self.epoch_threshold)

    def get_encoded_labels(self, mode: str = "three_class") -> np.ndarray:
        """Return integer-encoded labels for all epochs and channels.

        Args:
            mode : "three_class" (0, 1, 2) or "binary" (0, 1).
                   In binary mode, Slowing Waves and Spike/Sharp Waves are merged
                   into a single "Abnormal" class.

        Returns:
            Encoded label array of shape (n_epochs, n_channels).
        """
        if mode == "binary":
            return encode_labels_binary(self.labels)
        return self.labels

    def get_string_labels(self, mode: str = "three_class") -> np.ndarray:
        """Return string labels for all epochs and channels.

        Args:
            mode : "three_class" or "binary".

        Returns:
            String label array of shape (n_epochs, n_channels).
        """
        labels = np.empty_like(self.labels, dtype=object)
        labels[self.labels == 0] = "Normal"
        if mode == "binary":
            labels[self.labels == 1] = "Abnormal"
            labels[self.labels == 2] = "Abnormal"
        else:
            labels[self.labels == 1] = "Slowing Waves"
            labels[self.labels == 2] = "Spike and Sharp waves"
        return labels

    def free_memory(self) -> None:
        """Release accumulated memory."""
        gc.collect()


def compute_cwt(
    signal: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "mexh",
    method: str = "conv",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Continuous Wavelet Transform.

    Args:
        signal : 1D signal array.
        scales : Array of wavelet scales.
        wavelet: Wavelet name (default: 'mexh').
        method : Computation method, 'conv' or 'fft' (default: 'conv').

    Returns:
        coeffs : CWT coefficients array.
        freqs  : Corresponding frequencies (if available from pywt).
    """
    import pywt

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, method=method)
    return coeffs, freqs


class CWTProcessor:
    """Generate and save CWT scalogram images from EEG epochs.

    Uses multiple wavelet types to compute scalograms for each EEG channel.
    Scalograms are saved as PNG images using the 'nipy_spectral' colormap.

    Parameters
    ----------
    wavelet_types : List[str]
        Wavelet families to use (default: WAVELET_TYPES).
    scales       : np.ndarray
        CWT scales (default: SCALES).
    output_dir   : str | Path
        Root directory for saving images (default: './scalograms').
    colormap     : str
        Matplotlib colormap for scalogram visualization
        (default: 'nipy_spectral').
    extent       : tuple
        Scalogram extent [xmin, xmax, ymin, ymax] for imshow
        (default: [1, 31, 31, 1]).

    Attributes
    ----------
    wavelet_types : List[str]
    scales       : np.ndarray
    output_dir   : Path

    Example
    -------
    >>> processor = CWTProcessor(output_dir="./output")
    >>> processor.process_single_epoch(
    ...     epoch_data=np.random.randn(19, 400),
    ...     channel_idx=0,
    ...     epoch_idx=0,
    ...     file_id="0001",
    ...     label="Slowing Waves",
    ... )
    """

    def __init__(
        self,
        wavelet_types: List[str] = WAVELET_TYPES,
        scales: np.ndarray = SCALES,
        output_dir: Union[str, Path] = "./scalograms",
        colormap: str = "nipy_spectral",
        extent: Tuple[float, float, float, float] = (1, 31, 31, 1),
    ) -> None:
        self.wavelet_types = wavelet_types
        self.scales = scales
        self.output_dir = Path(output_dir)
        self.colormap = colormap
        self.extent = extent

        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["figure.figsize"] = [224 / 100, 224 / 100]

    def compute(
        self,
        signal: np.ndarray,
        wavelet: str = "mexh",
    ) -> np.ndarray:
        """Compute CWT coefficients for a single channel signal.

        Args:
            signal : 1D array of shape (n_samples,).
            wavelet: Wavelet type (default: first in wavelet_types).

        Returns:
            CWT coefficient array of shape (n_scales, n_samples).
        """
        coeffs, _ = compute_cwt(signal, self.scales, wavelet)
        return coeffs

    def _get_normalization_range(
        self, full_signal: np.ndarray, wavelet: str
    ) -> Tuple[float, float]:
        coef, _ = compute_cwt(full_signal, self.scales, wavelet)
        vmax = abs(coef).max()
        return -vmax, vmax

    def save_scalogram(
        self,
        coeffs: np.ndarray,
        output_path: Union[str, Path],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Save a CWT scalogram as a PNG image.

        Args:
            coeffs     : CWT coefficient array.
            output_path: Destination file path.
            vmin       : Minimum value for colormap normalization.
            vmax       : Maximum value for colormap normalization.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.imshow(
            coeffs,
            extent=self.extent,
            cmap=self.colormap,
            vmax=vmax,
            vmin=vmin,
        )
        plt.axis("off")
        plt.savefig(fname=str(output_path), bbox_inches="tight")
        plt.clf()
        plt.close()

    def process_single_channel(
        self,
        epoch_data: np.ndarray,
        channel_idx: int,
        epoch_idx: int,
        file_id: str,
        label: Union[int, str],
        split: str = "train",
        wavelet: Optional[str] = None,
    ) -> None:
        """Generate and save scalograms for one channel of one epoch.

        Args:
            epoch_data : EEG epoch array of shape (n_channels, window_size).
            channel_idx: Index of the channel to process.
            epoch_idx  : Index of the epoch for naming.
            file_id    : EDF file identifier for naming.
            label      : Integer (0/1/2) or string label.
            split      : Dataset split name ('train'/'valid'/'test').
            wavelet    : Wavelet type; if None, processes all in wavelet_types.
        """
        if isinstance(label, int):
            label_names = {0: "Normal", 1: "Slowing Waves", 2: "Spike and Sharp waves"}
            label_name = label_names.get(label, "Normal")
        else:
            label_name = label

        wavelets_to_use = [wavelet] if wavelet else self.wavelet_types

        for wtype in wavelets_to_use:
            filename = f"img_{epoch_idx}_{channel_idx}_{file_id}.png"
            dest = self.output_dir / wtype / split / label_name / filename

            if dest.exists():
                continue

            signal = epoch_data[channel_idx]
            coeffs = self.compute(signal, wavelet=wtype)

            if label_name == "Normal":
                vmin, vmax = None, None
            else:
                vmin, vmax = self._get_normalization_range(signal, wtype)

            self.save_scalogram(coeffs, dest, vmin=vmin, vmax=vmax)

    def process_epochs(
        self,
        epochs: List[np.ndarray],
        labels: np.ndarray,
        file_id: str,
        split: str = "train",
        max_normal_ratio: Optional[float] = None,
    ) -> None:
        """Process multiple epochs and save scalograms.

        For abnormal epochs (labels 1 or 2), all 19 channels are saved.
        For normal epochs, images are saved at regular intervals
        (every 30th epoch by default) to balance the dataset.

        Args:
            epochs      : List of epoch arrays, each (n_channels, window_size).
            labels      : Encoded labels array (n_epochs, n_channels).
            file_id     : EDF file identifier for naming.
            split       : Dataset split name.
            max_normal_ratio : If set, only save every Nth normal epoch.
        """
        n_epochs = len(epochs)
        for ep_idx in range(n_epochs):
            epoch_labels = labels[ep_idx]

            unique_labels = np.unique(epoch_labels)
            is_abnormal = any(lbl in [1, 2] for lbl in unique_labels)

            if is_abnormal:
                for ch_idx in range(epochs[ep_idx].shape[0]):
                    self.process_single_channel(
                        epochs[ep_idx],
                        ch_idx,
                        ep_idx,
                        file_id,
                        epoch_labels[ch_idx],
                        split=split,
                    )
            else:
                if max_normal_ratio is not None and ep_idx % int(max_normal_ratio) != 0:
                    continue
                for ch_idx in range(epochs[ep_idx].shape[0]):
                    self.process_single_channel(
                        epochs[ep_idx],
                        ch_idx,
                        ep_idx,
                        file_id,
                        0,
                        split=split,
                    )

    def create_output_directories(
        self,
        splits: Optional[List[str]] = None,
    ) -> None:
        """Create the output directory structure for all splits and labels.

        Args:
            splits : List of split names (default: ['train', 'valid', 'test']).
        """
        if splits is None:
            splits = ["train", "valid", "test"]

        for wavelet in self.wavelet_types:
            for split in splits:
                for label in DEST_LIST:
                    (self.output_dir / wavelet / split / label).mkdir(
                        parents=True, exist_ok=True
                    )

    def free_memory(self) -> None:
        """Release matplotlib memory."""
        plt.clf()
        plt.close("all")
        gc.collect()
