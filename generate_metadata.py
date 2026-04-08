"""
generate_metadata.py
--------------------
Extracts subject metadata (gender, age) from CSV annotation files and saves to CSV.
The CSV files contain gender and age in the header row.
"""

import os
import glob
import pandas as pd
import re


def extract_metadata_from_csv(csv_path):
    """
    Extract subject_id, gender, and age from CSV annotation file.

    Returns dict with keys: subject_id, gender, age
    """
    try:
        # Read first few lines to get header info
        df = pd.read_csv(csv_path, nrows=1)

        # Get first row values
        gender = df["Gender"].values[0] if "Gender" in df.columns else "unknown"
        age_str = df["Age"].values[0] if "Age" in df.columns else "-1"

        # Extract age number from string like "8 years"
        age = -1
        if pd.notna(age_str) and age_str:
            age_match = re.search(r"(\d+)", str(age_str))
            if age_match:
                age = int(age_match.group(1))

        # Extract subject_id from filename (e.g., "4.csv" -> "0000004")
        filename = os.path.basename(csv_path)
        subject_id_num = os.path.splitext(filename)[0]
        # Pad with zeros to match EDF format
        subject_id = subject_id_num.zfill(7)

        # Normalize gender
        gender_str = (
            str(gender).strip().lower()
            if gender and str(gender) != "nan"
            else "unknown"
        )
        if gender_str in ["m", "male"]:
            gender_normalized = "Male"
        elif gender_str in ["f", "female"]:
            gender_normalized = "Female"
        else:
            gender_normalized = "unknown"

        return {"subject_id": subject_id, "gender": gender_normalized, "age": age}

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def generate_metadata_csv(csv_dir, output_csv):
    """
    Generate metadata CSV from CSV annotation directory.

    Args:
        csv_dir: Directory containing CSV annotation files
        output_csv: Output CSV path
    """
    all_metadata = []

    # Find all CSV files
    csv_pattern = os.path.join(csv_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    print(f"Processing {len(csv_files)} CSV files from {csv_dir}")

    for csv_file in csv_files:
        metadata = extract_metadata_from_csv(csv_file)
        if metadata:
            all_metadata.append(metadata)
            print(
                f"  Extracted: {metadata['subject_id']} - {metadata['gender']}, {metadata['age']}"
            )

    # Create DataFrame and save
    df = pd.DataFrame(all_metadata)
    df = df.drop_duplicates(subset=["subject_id"])
    df = df.sort_values("subject_id")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nMetadata saved to {output_csv}")
    print(f"Total subjects: {len(df)}")
    print(f"Gender distribution:\n{df['gender'].value_counts()}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")

    return df


def main():
    # CSV directory (for abnormal subjects)
    csv_dir = "eeg_data/Data/NMT_Events/csv/SW & SSW CSV Files"

    output_csv = "data/nmt_metadata.csv"

    print("=" * 60)
    print("Generating Metadata CSV from CSV annotation files")
    print("=" * 60)

    generate_metadata_csv(csv_dir, output_csv)


if __name__ == "__main__":
    main()
