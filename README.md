# EEG Anomaly Triage

A PyTorch-based deep learning pipeline for EEG anomaly classification using CWT scalogram images. Supports both **three-class** (Normal, Slowing Waves, Spike/Sharp Waves) and **binary** (Normal vs Abnormal) classification modes.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Window Extraction](#window-extraction-windowextractor)
   - [Label Assignment Flow](#label-assignment-flow)
   - [CWT Scalogram Generation](#cwt-scalogram-generation-cwtprocessor)
5. [Training Pipeline](#training-pipeline)
6. [Label Encoding System](#label-encoding-system)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [Models](#models)
10. [End-to-End Workflow](#end-to-end-workflow)
11. [Collaboration Analysis](#collaboration-analysis)

---

## Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EEG ANOMALY TRIAGE PIPELINE                         │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │  RAW EDF    │         │  WINDOW     │         │     CWT     │
    │  FILES      │ ──────→ │  EXTRACTION │ ──────→ │  SCALOGRAM  │
    │  (19 ch)    │         │  + LABELS    │         │   IMAGES    │
    └─────────────┘         └─────────────┘         └─────────────┘
                                                            │
                                                            ▼
    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
    │   TRAINED   │ ←────── │  TRAINING   │ ←────── │   PyTorch   │
    │   MODEL     │         │   LOOP      │         │   DATASET   │
    └─────────────┘         └─────────────┘         └─────────────┘
```

This pipeline:

1. **Reads raw EDF EEG files** and CSV annotations
2. **Splits data into 50% overlapping 2-second windows** (400 samples at 200 Hz)
3. **Assigns labels** to each window based on annotation timestamps
4. **Computes Continuous Wavelet Transform (CWT)** scalograms
5. **Trains CNN models** (VGG16, GoogLeNet, EfficientNetB1) on the scalogram images
6. **Supports two classification modes**: three-class or binary

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              STEP 1: PREPROCESSING                            │
│                          (EDF Files → Scalogram Images)                        │
└──────────────────────────────────────────────────────────────────────────────┘

    RAW EDF FILES                          CSV ANNOTATIONS
         │                                        │
         │  mne.io.read_raw_edf()                 │  pd.read_csv()
         ▼                                        ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         WindowExtractor                               │
    │                                                                       │
    │  1. Load EEG data: Shape (19, N_samples)                             │
    │                                                                       │
    │  2. Split into windows: 50% overlap, 400 samples each                 │
    │     → List of [19, 400] arrays                                       │
    │                                                                       │
    │  3. Parse CSV annotations:                                          │
    │     • extract_labels_from_csv()                                      │
    │     • clean_labels() → map to super-classes                          │
    │     • generate_label_array() → per-channel labels                    │
    │                                                                       │
    │  4. Assign labels per window:                                       │
    │     • threshold_epoch_labels() → 25% threshold                       │
    │     • encode_labels() → integers                                     │
    │                                                                       │
    │  Output:                                                              │
    │    epochs  → List[[19, 400]]                                         │
    │    labels  → [n_epochs, 19] with values 0, 1, 2                      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        CWTProcessor                                   │
    │                                                                       │
    │  For each epoch and channel:                                         │
    │    compute_cwt(signal, scales=[1..23], wavelet)                       │
    │                                                                       │
    │  Wavelets: mexh, morl, gaus1, gaus2                                 │
    │  Colormap: nipy_spectral                                            │
    │                                                                       │
    │  Save as PNG: 224×224 pixels                                         │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌─────────────────────────────────────────────────────────────────────┐
     │                      IMAGE FOLDER STRUCTURE                           │
     │                                                                       │
     │  data/                                                               │
     │  ├── nmt/                                                            │
     │  │   ├── train/                                                      │
     │  │   │   ├── Normal/                                                │
     │  │   │   │   ├── nmt_0000001_0_Normal.png                          │
     │  │   │   │   └── ...                                                │
     │  │   │   ├── Slowing Waves/                                         │
     │  │   │   └── Spike and Sharp waves/                                 │
     │  │   ├── valid/                                                     │
     │  │   └── test/                                                      │
     │                                                                       │
     │  NOTE: Same images for both modes! Labels are encoded on-the-fly.    │
     └─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
EEG_Anomaly_Triage/
├── configs/
│   ├── dataset_config.yaml    # Dataset definitions with subject splits
│   └── training.yaml         # Training configuration (mode, model, etc.)
├── utils/
│   ├── __init__.py           # Exports all public utilities
│   ├── mne_dataset.py        # Dataset generation from EDF/CSV
│   ├── DNNs.py              # VGG16, GoogLeNet, EfficientNetB1 (nn.Module)
│   ├── preprocessing.py      # WindowExtractor, CWTProcessor, label functions
│   └── experiment_recorder.py
├── train.py                  # Training and preprocessing script
├── checkpoints/              # Trained model checkpoints
│   ├── nmt_vgg16_three_class_10.pt
│   └── ...
├── data/                    # Scalogram images (generated by --preprocess)
│   └── nmt/
│       ├── train/
│       │   ├── Normal/
│       │   ├── Slowing Waves/
│       │   └── Spike and Sharp waves/
│       ├── valid/
│       └── test/
├── experiments/              # Collaboration analysis results
│   └── experiment.yaml      # All experiment runs (timestamp-keyed)
└── artifacts/
    └── source_code_files/   # Original Jupyter notebooks (reference)
```

---

## Preprocessing Pipeline

### Window Extraction (`WindowExtractor`)

The `WindowExtractor` class handles EDF file reading, windowing, and label assignment:

```python
from utils import WindowExtractor

extractor = WindowExtractor()
epochs, labels = extractor.process(
    edf_path="data/0001.edf",
    csv_path="annotations/"
)
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 200 Hz | EEG sampling frequency |
| `window_size` | 400 samples | 2 seconds at 200 Hz |
| `overlap` | 0.5 | 50% overlap between windows |
| `min_ab_threshold` | 0.7 | Min fraction of annotation to affect window |
| `epoch_threshold` | 25.0 | % threshold for assigning epoch label |

**Output:**
- `epochs`: `List[np.ndarray]` of shape `[19, 400]` each
- `labels`: `np.ndarray` of shape `[n_epochs, 19]` with values 0, 1, 2

**Methods:**
| Method | Description |
|--------|-------------|
| `process(edf_path, csv_path)` | Load EDF/CSV, extract windows and labels |
| `get_encoded_labels(mode)` | Return integer labels (mode: `"three_class"` or `"binary"`) |
| `get_string_labels(mode)` | Return string labels |
| `free_memory()` | Release accumulated memory |

---

### Label Assignment Flow

Windows are labeled based on neurologist annotations in the CSV files. Each 2-second window gets a label for each of the 19 EEG channels.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     LABEL ASSIGNMENT FLOW                                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 1: CSV Annotations (from neurologist)                                 │
│  ───────────────────────────────────────────────────────────────────────   │
│  Each row contains:                                                         │
│    • Start time : "00:15:30.500"                                          │
│    • End time   : "00:15:35.200"                                          │
│    • Channel    : "FP1 F3 F7"                                             │
│    • Comment    : "generalized delta slow waves"                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  extract_labels_from_csv()
                                    │  (converts HH:MM:SS.mmm → sample indices × 200 Hz)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 2: Sample Indices                                                    │
│  ───────────────────────────────────────────────────────────────────────   │
│  Sampling rate: 200 Hz                                                      │
│                                                                             │
│  Start: (0h × 3600 + 15m × 60 + 30s + 0.5s) × 200 = sample 186100       │
│  End  : (0h × 3600 + 15m × 60 + 35s + 0.2s) × 200 = sample 187040       │
│                                                                             │
│  Window size: 400 samples = 2 seconds                                      │
│  Step size: 200 samples = 1 second (50% overlap)                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  clean_labels() + generate_label_array()
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 3: Per-Channel Labels                                                │
│  ───────────────────────────────────────────────────────────────────────   │
│  Annotation affects channels: FP1, F3, F7                                   │
│                                                                             │
│  Channel Labels Array (for this annotation event):                          │
│  ┌──────┬──────┬──────┬──────┬──────┬─────┬──────┬──────┐                │
│  │ FP1  │ FP2  │ F3   │ F4   │ C3   │ ... │ FZ   │ PZ   │ CZ            │
│  ├──────┼──────┼──────┼──────┼──────┼─────┼──────┼──────┤                │
│  │ Slow │ Norm │ Slow │ Norm │ Norm │ ... │ Norm │ Norm │ Norm           │
│  │ Wave │ al   │ Wave │ al   │ al   │     │ al   │ al   │               │
│  └──────┴──────┴──────┴──────┴──────┴─────┴──────┴──────┘                │
│                                                                             │
│  "Slow Wave" is mapped from "delta slow waves" via clean_labels()          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  WindowExtractor broadcasts to windows
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 4: Window Broadcasting                                               │
│  ───────────────────────────────────────────────────────────────────────   │
│  Annotation duration: 5 seconds (samples 186100 → 187040)                   │
│  Window size: 2 seconds, Step: 1 second                                    │
│                                                                             │
│  Sample 186100 → Window index: 186100 / 400 = 465.25                      │
│  Sample 187040 → Window index: 187040 / 400 = 467.6                        │
│                                                                             │
│  Annotation spans windows: 465, 466, 467                                    │
│                                                                             │
│  ┌──────────┬────────────────────────────────────────────────────────────┐  │
│  │ Window   │ Affected Channels                                           │  │
│  ├──────────┼────────────────────────────────────────────────────────────┤  │
│  │ 465      │ FP1, F3, F7 → "Delta Slow Wave"                           │  │
│  │ 466      │ FP1, F3, F7 → "Delta Slow Wave"                           │  │
│  │ 467      │ FP1, F3, F7 → "Delta Slow Wave"                           │  │
│  │ Other    │ All channels → "Normal"                                    │  │
│  └──────────┴────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  threshold_epoch_labels() with 25% threshold
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Step 5: Final Labels [n_epochs, 19]                                        │
│  ───────────────────────────────────────────────────────────────────────   │
│  A window gets a non-zero label only if ≥25% of its 400 samples overlap     │
│  with a neurologist-annotated region.                                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Window 465, Channel FP1:                                               │   │
│  │   • Samples 186000-186399 (window 465)                                │   │
│  │   • Annotation: samples 186100-187040                                  │   │
│  │   • Overlap: samples 186100-186399 = 300 samples                       │   │
│  │   • Percentage: 300/400 = 75% > 25% threshold                         │   │
│  │   • → Label = 1 (Delta Slow Wave)                                     │   │
│  │                                                                        │   │
│  │ Window 463, Channel FP1:                                               │   │
│  │   • Samples 185600-185999 (window 463)                                │   │
│  │   • Annotation: samples 186100-187040                                  │   │
│  │   • Overlap: samples 186100-185999 = 0 samples                        │   │
│  │   • → Label = 0 (Normal)                                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Key Parameters Explained

| Parameter | Value | Effect |
|-----------|-------|--------|
| `min_ab_threshold` | 0.7 | A window is affected by an annotation only if at least 70% of the window falls within the annotated region |
| `epoch_threshold` | 25.0% | A window is labeled as abnormal only if ≥25% of its samples belong to an annotated region |

#### Label Assignment Example

Given a neurologist annotation: `"00:15:30-00:15:35"`, channels `FP1 F3 F7`, label `"delta slow waves"`:

| Window | Start Time | End Time | Overlap with Annotation | FP1 Label | F3 Label | F7 Label |
|--------|-----------|----------|------------------------|-----------|----------|----------|
| 465 | 00:15:30.0 | 00:15:32.0 | 100% | 1 (Slow Wave) | 1 (Slow Wave) | 1 (Slow Wave) |
| 466 | 00:15:31.0 | 00:15:33.0 | 100% | 1 (Slow Wave) | 1 (Slow Wave) | 1 (Slow Wave) |
| 467 | 00:15:32.0 | 00:15:34.0 | 60% | 1 (Slow Wave) | 1 (Slow Wave) | 1 (Slow Wave) |
| 468 | 00:15:33.0 | 00:15:35.0 | 40% | 0 (Normal) | 0 (Normal) | 0 (Normal) |
| 469 | 00:15:34.0 | 00:15:36.0 | 20% | 0 (Normal) | 0 (Normal) | 0 (Normal) |

The 25% threshold means windows 465-467 (with ≥25% overlap) are labeled as abnormal, while windows 468-469 (with <25% overlap) remain Normal.

---

### CWT Scalogram Generation (`CWTProcessor`)

```python
from utils import CWTProcessor

processor = CWTProcessor(output_dir="./data")
processor.create_output_directories()
processor.process_epochs(
    epochs=epochs,
    labels=labels,
    file_id="0001",
    split="train"
)
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `wavelet_types` | `["mexh", "morl", "gaus1", "gaus2"]` | Wavelet families |
| `scales` | `np.arange(1, 24)` | CWT scales |
| `output_dir` | `"./scalograms"` | Root directory for images |
| `colormap` | `"nipy_spectral"` | Visualization colormap |

**Image Naming Convention:**
```
img_{epoch_idx}_{channel_idx}_{file_id}.png
```

Example: `img_5_3_0001.png` → Epoch 5, Channel 3, from EDF file 0001

---

## Training Pipeline

### Configuration (`configs/training.yaml`)

```yaml
training:
  mode: three_class          # "three_class" | "binary"
  model: vgg16              # "vgg16" | "googlenet" | "efficientnetb1"
  num_classes: 3            # Auto-derived from mode (3 or 2)
  learning_rate: 0.0001
  epochs: 30
  batch_size: 32
  checkpoint_dir: checkpoints
  early_stopping:
    patience: 5
    min_delta: 0.001
  data:
    train_dir: data/train
    valid_dir: data/valid
    test_dir: data/test
```

### Training Script

```bash
# Use config defaults
python train.py

# Override via CLI
python train.py --mode binary --model googlenet --epochs 20

# Custom config file
python train.py --config configs/my_config.yaml
```

### Label Mapping in Dataset

The `EEGCWTDataset` class automatically maps folder names to labels based on mode:

| Folder Name | `three_class` | `binary` |
|-------------|----------------|----------|
| `Normal` | 0 | 0 |
| `Slowing Waves` | 1 | 1 (Abnormal) |
| `Spike and Sharp waves` | 2 | 1 (Abnormal) |

---

## Label Encoding System

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LABEL ENCODING SYSTEM                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Raw CSV Annotations                                               │
│  ─────────────────────────────────                                           │
│  Comment column values (hundreds of variations):                             │
│    • "slow waves"                                                           │
│    • "generalized sharp waves discharge"                                    │
│    • "delta waves"                                                          │
│    • "spike and wave discharge"                                            │
│    • "2 hertz spike and wave discharge"                                    │
│    • etc.                                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  clean_labels()
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Super-class Labels (4 categories)                                 │
│  ─────────────────────────────────────────────                              │
│  • "Normal"                                                                 │
│  • "Delta Slow Wave"                                                        │
│  • "Sharp Wave"                                                             │
│  • "Spike and Wave Discharge"                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  generate_label_array() + WindowExtractor
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Epoch Labels [n_epochs, 19]                                      │
│  ────────────────────────────────────────                                   │
│  Values per cell: "Normal", "Delta Slow Wave", "Sharp Wave", etc.          │
│  Encoded as integers via threshold_epoch_labels():                           │
│    • Count non-zero labels in each 400-sample window                        │
│    • If > 25% of window has abnormality → assign that label                  │
│    • Otherwise → 0 (Normal)                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  encode_labels() or encode_labels_binary()
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 4: Final Integer Encoding                                            │
│  ────────────────────────────────                                           │
│                                                                             │
│  encode_labels() → three_class mode:                                        │
│  ┌────────────────────────────────┬───────────────────────────────────┐    │
│  │ String Label                   │ Integer                           │    │
│  ├────────────────────────────────┼───────────────────────────────────┤    │
│  │ "Normal"                      │ 0                                 │    │
│  │ "Delta Slow Wave"             │ 1                                 │    │
│  │ "Sharp Wave"                  │ 2                                 │    │
│  │ "Spike and Wave Discharge"    │ 2                                 │    │
│  └────────────────────────────────┴───────────────────────────────────┘    │
│                                                                             │
│  encode_labels_binary() → binary mode:                                       │
│  ┌────────────────────────────────┬───────────────────────────────────┐    │
│  │ String Label                   │ Integer                           │    │
│  ├────────────────────────────────┼───────────────────────────────────┤    │
│  │ "Normal"                      │ 0                                 │    │
│  │ "Delta Slow Wave"             │ 1 (Abnormal)                     │    │
│  │ "Sharp Wave"                  │ 1 (Abnormal)                     │    │
│  │ "Spike and Wave Discharge"   │ 1 (Abnormal)                     │    │
│  └────────────────────────────────┴───────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Dataset Configuration (`configs/dataset_config.yaml`)

Each dataset must specify explicit train/valid/test subject IDs:

```yaml
datasets:
  nmt:
    braindecode_class: NMT
    path: eeg_data/Data/NMT_Events
    target_name: pathological
    preload: false
    train_subject_ids: [1, 2, 3, 4, 5, 6]
    valid_subject_ids: [7, 8]
    test_subject_ids: [9, 10]
    n_jobs: 1
```

**Note:** All datasets must have `train_subject_ids`, `valid_subject_ids`, and `test_subject_ids`. These must not overlap.

### Training Configuration (`configs/training.yaml`)

```yaml
training:
  mode: three_class          # "three_class" | "binary"
  model: vgg16              # "vgg16" | "googlenet" | "efficientnetb1"
  num_classes: 3            # Auto-derived from mode (3 or 2)
  learning_rate: 0.0001
  epochs: 30
  batch_size: 32
  checkpoint_dir: checkpoints
  early_stopping:
    patience: 5
    min_delta: 0.001
```

### Model Selection

Available models in `configs/training.yaml`:

| Model | Description | Pretrained |
|-------|-------------|------------|
| `vgg16` | Standard VGG-16 (13 conv + 3 FC layers) | No |
| `googlenet` | Custom Inception network with auxiliary classifiers | No |
| `efficientnetb1` | EfficientNet-B1 from timm | Yes (ImageNet) |

---

## Usage

### Generate Scalogram Images

First, generate the CWT scalogram images from raw EDF/CSV files:

```bash
python train.py --preprocess
```

This reads `configs/dataset_config.yaml` and generates images for all configured datasets into:
- `data/{dataset_name}/train/`
- `data/{dataset_name}/valid/`
- `data/{dataset_name}/test/`

**Note:** The `--dataset` name in commands maps directly to the subdirectory under `data/`. For example, `--dataset nmt` expects data in `data/nmt/train/` and `data/nmt/valid/`.

### Train a Model

```bash
# Train on specific dataset (data/nmt/train and data/nmt/valid must exist)
python train.py --dataset nmt

# With custom settings
python train.py --dataset nmt --mode binary --model googlenet --epochs 20
```

**Available options:**
| Flag | Description | Values |
|------|-------------|--------|
| `--dataset` | Dataset name from dataset_config.yaml | `nmt`, etc. |
| `--mode` | Classification mode | `three_class`, `binary` |
| `--model` | Model architecture | `vgg16`, `googlenet`, `efficientnetb1` |
| `--epochs` | Number of training epochs | Integer |

### Load and Use Trained Model

```python
import torch
from utils import VGG16, GoogLeNet, EfficientNetB1

# Create model (must match training configuration)
num_classes = 3  # or 2 for binary
model = VGG16(num_classes=num_classes)

# Load checkpoint (use dataset name if training with --dataset)
checkpoint = torch.load("checkpoints/nmt_vgg16_three_class_best.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    # images: (B, 3, 224, 224)
    output = model(images)
    predictions = output.argmax(dim=1)
```

### Run Collaboration Analysis

Evaluate Human-AI collaboration strategies on trained models:

```bash
# Default settings
python test_and_collab.py

# Custom settings
python test_and_collab.py --dataset nmt --confidence-threshold 0.9 --cost-alpha 0.1
```

**Available options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Dataset name | `nmt` |
| `--confidence-threshold` | Strategy A: defer if confidence below this | `0.85` |
| `--cost-alpha` | Strategy B: defer if P(pathology) > this | `0.15` |

Results are saved to `experiments/experiment.yaml` with timestamp keys.

---

## Models

### VGG16

Standard VGG-16 architecture translated from Keras implementation.

```
Input: (B, 3, 224, 224)
├── block1: Conv64×2 → MaxPool
├── block2: Conv128×2 → MaxPool
├── block3: Conv256×3 → MaxPool
├── block4: Conv512×3 → MaxPool
├── block5: Conv512×3 → MaxPool
├── Flatten
├── FC: 4096 → Dropout(0.5)
├── FC: 4096 → Dropout(0.5)
└── FC: num_classes
Output: (B, num_classes)
```

### GoogLeNet

Custom Inception-v1 implementation with auxiliary classifiers.

```
Input: (B, 3, 224, 224)
├── conv1: Conv64, 7×7 → MaxPool
├── conv2: Conv64, 1×1 → Conv192, 3×3 → MaxPool
├── inception3a, 3b
├── inception4a ──────────────────────────────────┐ → aux1 (0.3×loss)
├── inception4b, 4c, 4d, 4e → MaxPool             │
├── inception5a, 5b → GAP                          │
└── FC: num_classes ───────────────────────────────┘ → aux2 (0.3×loss)

Total loss = main_loss + 0.3 × aux1_loss + 0.3 × aux2_loss
Output: (B, num_classes)
```

### EfficientNetB1

Pre-trained EfficientNet-B1 with custom classifier head.

```
Input: (B, 3, 224, 224)
├── EfficientNet-B1 backbone (ImageNet pretrained)
├── Dropout(0.4)
└── FC: num_classes
Output: (B, num_classes)
```

---

## End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING (Run once to generate images)                              │
│ ────────────────────────────────────────────────────                         │
│                                                                              │
│ from utils import WindowExtractor, CWTProcessor                               │
│                                                                              │
│ extractor = WindowExtractor()                                                 │
│ epochs, labels = extractor.process("edf/0001.edf", "annotations/")           │
│                                                                              │
│ processor = CWTProcessor(output_dir="./data")                                │
│ processor.create_output_directories()                                         │
│ processor.process_epochs(epochs, labels, file_id="0001", split="train")     │
│                                                                              │
│ # Repeat for all EDF files and splits (train/valid/test)                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. TRAINING (Run after images are generated)                                │
│ ────────────────────────────────────────────────────                         │
│                                                                              │
│ # Edit configs/training.yaml:                                               │
│ #   mode: three_class  (or binary)                                          │
│ #   model: vgg16       (or googlenet, efficientnetb1)                        │
│                                                                              │
│ python train.py                                                             │
│                                                                              │
│ # Checkpoints saved to: checkpoints/                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. INFERENCE (After training)                                                │
│ ────────────────────────────────                                            │
│                                                                              │
│ import torch                                                                 │
│ from utils import VGG16                                                       │
│                                                                              │
│ model = VGG16(num_classes=3)                                                 │
│ checkpoint = torch.load("checkpoints/nmt_vgg16_three_class_best.pt")          │
│ model.load_state_dict(checkpoint['model_state_dict'])                        │
│ model.eval()                                                                 │
│                                                                              │
│ with torch.no_grad():                                                        │
│     output = model(images)  # (B, num_classes)                              │
│     predictions = output.argmax(dim=1)                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Collaboration Analysis

Script: `test_and_collab.py`

Evaluates Human-AI collaboration strategies for clinical decision support.

### Collaboration Strategies

| Strategy | Description |
|----------|-------------|
| **Baseline (AI Alone)** | Standard model predictions without collaboration |
| **Strategy A** (Confidence-based) | If confidence < threshold, defer to human |
| **Strategy B** (Cost-aware) | If P(Pathology) > α, defer to human |

### Usage

```bash
# Default settings
python test_and_collab.py

# Custom settings
python test_and_collab.py --dataset nmt --confidence-threshold 0.9 --cost-alpha 0.1
```

### Output

Results saved to `experiments/experiment.yaml` with timestamp keys:

```yaml
runs:
  "2026-04-08_14-30-00":
    model:
      name: vgg16
      dataset: nmt
      mode: binary
      num_classes: 2
    collaboration:
      strategy_a:
        confidence_threshold: 0.85
      strategy_b:
        cost_alpha: 0.15
    results:
      baseline:
        accuracy: 0.85
        precision: 0.83
        recall: 0.87
        f1: 0.85
      strategy_a:
        accuracy: 0.89
        precision: 0.88
        recall: 0.92
        f1: 0.90
        escalation_rate: 15.0
      strategy_b:
        accuracy: 0.91
        precision: 0.90
        recall: 0.95
        f1: 0.92
        escalation_rate: 25.0
```

### Collaboration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.85 | Strategy A: defer if confidence below this |
| `cost_alpha` | 0.15 | Strategy B: defer if P(pathology) > this |

---

## Key Parameters Summary

| Component | Parameter | Value | Description |
|-----------|-----------|-------|-------------|
| **WindowExtractor** | `sampling_rate` | 200 Hz | EEG sampling frequency |
| | `window_size` | 400 samples | 2 seconds |
| | `overlap` | 0.5 | 50% overlap |
| | `epoch_threshold` | 25.0 | Label assignment threshold |
| **CWTProcessor** | `wavelets` | mexh, morl, gaus1, gaus2 | Wavelet families |
| | `scales` | 1-23 | CWT scale range |
| | `image_size` | 224×224 | Output image dimensions |
| **Training** | `learning_rate` | 0.0001 | Adam optimizer LR |
| | `batch_size` | 32 | Training batch size |
| | `patience` | 5 | Early stopping patience |

---

## Dependencies

```python
# Core
numpy
pandas
pyyaml

# Deep Learning
torch
torchvision
timm  # For EfficientNet pretrained weights

# EEG Processing
mne  # For EDF file reading

# Wavelet Transform
pywt  # PyWavelets for CWT

# Image Processing
matplotlib
PIL

# Data Loading
torch.utils.data.DataLoader
```

---

## Notes

- **Same images for both modes**: Scalogram images are generated once. The training script applies label encoding on-the-fly based on the selected mode.
- **Class weights**: The training script automatically computes class weights to handle imbalanced datasets.
- **Early stopping**: Training stops if validation loss doesn't improve by `min_delta` for `patience` epochs.
- **Checkpoint naming**: Only the best model is saved: `{dataset}_{model}_{mode}_best.pt` (e.g., `nmt_vgg16_three_class_best.pt`)
