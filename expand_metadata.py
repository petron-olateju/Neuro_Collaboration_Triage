"""
expand_metadata.py
-----------------
Expand metadata CSV to include all subjects from EDF files.
For subjects without annotations, use "unknown" for gender and -1 for age.
"""

import os
import re
import pandas as pd
from pathlib import Path


def get_metadata_from_csv_annotations(csv_dir):
    """Extract metadata from CSV annotation files."""
    metadata = {}

    for f in os.listdir(csv_dir):
        if not f.endswith(".csv"):
            continue

        filepath = os.path.join(csv_dir, f)
        try:
            df = pd.read_csv(filepath, nrows=1)

            # Get first row values
            gender_row = df.get("Gender", ["unknown"])[0]
            age_row = df.get("Age", ["-1"])[0]

            # Extract gender
            gender_str = str(gender_row).strip().lower() if gender_row else "unknown"
            if gender_str in ["m", "male"]:
                gender = "Male"
            elif gender_str in ["f", "female"]:
                gender = "Female"
            else:
                gender = "unknown"

            # Extract age
            age = -1
            if pd.notna(age_row) and age_row:
                age_match = re.search(r"(\d+)", str(age_row))
                if age_match:
                    age = int(age_match.group(1))

            # Subject ID from filename (keep as int for matching)
            subject_id = int(f.replace(".csv", ""))
            metadata[subject_id] = {"gender": gender, "age": age}

        except Exception as e:
            print(f"Error reading {f}: {e}")

    return metadata


def get_all_edf_subjects(edf_dirs):
    """Get all subject IDs from EDF files."""
    subjects = set()
    for edf_dir in edf_dirs:
        if not os.path.exists(edf_dir):
            continue
        for f in os.listdir(edf_dir):
            if f.endswith(".edf"):
                sid = int(f.replace(".edf", ""))
                subjects.add(sid)
    return subjects


def main():
    print("=" * 60)
    print("Expanding Metadata CSV")
    print("=" * 60)

    # Paths
    csv_dir = "eeg_data/Data/NMT_Events/csv/SW & SSW CSV Files"
    edf_normal_dir = "eeg_data/Data/NMT_Events/edf/Normal EDF Files"
    edf_abnormal_dir = "eeg_data/Data/NMT_Events/edf/Abnormal EDF Files"
    output_csv = "data/nmt_metadata.csv"

    # Step 1: Get metadata from CSV annotations
    print("\n1. Extracting metadata from CSV annotations...")
    csv_metadata = get_metadata_from_csv_annotations(csv_dir)
    print(f"   Found {len(csv_metadata)} subjects with annotations")

    # Step 2: Get all subjects from EDF files
    print("\n2. Scanning EDF files...")
    all_edf_subjects = get_all_edf_subjects([edf_normal_dir, edf_abnormal_dir])
    print(f"   Found {len(all_edf_subjects)} subjects in EDF files")

    # Step 3: Merge - prefer CSV annotations, fallback to EDF subjects
    print("\n3. Building complete metadata...")
    all_subjects = all_edf_subjects | set(csv_metadata.keys())

    metadata_rows = []
    for sid in sorted(all_subjects):
        if sid in csv_metadata:
            # Use CSV annotation
            row = {
                "subject_id": sid,  # Keep as int
                "gender": csv_metadata[sid]["gender"],
                "age": csv_metadata[sid]["age"],
            }
        else:
            # Use default for subjects without annotations
            row = {"subject_id": sid, "gender": "unknown", "age": -1}
        metadata_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(metadata_rows)
    df = df.sort_values("subject_id")

    # Save as integer format
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n4. Saved to {output_csv}")
    print(f"   Total subjects: {len(df)}")
    print(f"\n   Gender distribution:")
    print(df["gender"].value_counts().to_string())
    print(f"\n   Age range: {df['age'].min()} to {df['age'].max()}")

    # Verify which test subjects now have metadata
    print("\n5. Verifying test subjects...")
    test_dir = Path("data/nmt/test")
    test_subjects = set()
    for subdir in test_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix == ".png":
                    test_subjects.add(int(f.stem.split("_")[0]))

    test_with_meta = []
    test_without = []
    for sid in test_subjects:
        row = df[df["subject_id"] == sid]
        if not row.empty and row.iloc[0]["gender"] != "unknown":
            test_with_meta.append(sid)
        else:
            test_without.append(sid)

    print(f"   Test subjects: {sorted(test_subjects)}")
    print(f"   With metadata: {sorted(test_with_meta)}")
    print(f"   Missing: {sorted(test_without)}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
