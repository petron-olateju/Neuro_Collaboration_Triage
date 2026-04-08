"""
dataset_with_metadata.py
-------------------------
Dataset wrapper that adds metadata (subject_id, gender, age) to EEGCWTDataset.
"""

import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path


class EEGCWTMetadataDataset(Dataset):
    """
    Wraps EEGCWTDataset to return metadata alongside images/labels.

    Extracts subject_id from image filename and maps to gender/age from
    metadata CSV.

    Parameters
    ----------
    base_dataset : EEGCWTDataset
        The underlying EEGCWT dataset to wrap.
    metadata_csv : str
        Path to CSV file with columns: subject_id, gender, age
    """

    def __init__(self, base_dataset, metadata_csv):
        self.base_dataset = base_dataset
        self.metadata_df = pd.read_csv(metadata_csv)

        # Set subject_id as index (convert to int to handle both formats)
        if "subject_id" in self.metadata_df.columns:
            self.metadata_df["subject_id"] = self.metadata_df["subject_id"].astype(int)
            self.metadata_df = self.metadata_df.set_index("subject_id")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns tuple: (image, label, metadata_dict)
        """
        image, label = self.base_dataset[idx]

        # Extract subject_id from filename
        # Filename format: "0000025_0_Normal" -> subject_id: "0000025"
        img_path = self.base_dataset.samples[idx][0]
        filename = img_path.stem  # e.g., "0000025_0_Normal"
        parts = filename.split("_")
        subject_id_str = parts[0] if parts else "unknown"

        # Convert to int for lookup (CSV has int index, e.g., 25 not "0000025")
        try:
            subject_id_int = int(subject_id_str)
        except ValueError:
            subject_id_int = -1

        # Get metadata from CSV - handle both int and string index
        try:
            if subject_id_int in self.metadata_df.index:
                meta = self.metadata_df.loc[subject_id_int]
            else:
                meta = None
        except KeyError:
            meta = None

        if meta is not None and not pd.isna(meta).all():
            gender = str(meta.get("gender", "unknown"))
            age = int(meta.get("age", -1))
        else:
            gender = "unknown"
            age = -1

        metadata = {"subject_id": subject_id_str, "gender": gender, "age": age}

        return image, label, metadata


def classify_age_group(age):
    """
    Classify age into groups.

    Parameters
    ----------
    age : int
        Age in years

    Returns
    -------
    str : Age group category
    """
    if age < 0:
        return "unknown"
    elif age <= 18:
        return "pediatric"
    elif age <= 64:
        return "adult"
    else:
        return "senior"
