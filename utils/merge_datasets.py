"""
Merge original spectrograms + augmented spectrograms into one unified dataset.
Copies ALL .npy files into data/final_dataset/data/ and updates paths in CSV.

Input:
  - data/processed/spectrograms/processed_metadata.csv  (original .npy)
  - data/metadata/augmented_metadata.csv                (augmented .npy)

Output:
  - data/final_dataset/data/   (all .npy files)
  - data/final_dataset/final_dataset.csv
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm

ORIG_CSV   = "data/processed/spectrograms/processed_metadata.csv"
AUG_CSV    = "data/metadata/augmented_metadata.csv"
OUTPUT_DIR = "data/final_dataset"
DATA_DIR   = os.path.join(OUTPUT_DIR, "data").replace("\\", "/")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_dataset.csv").replace("\\", "/")

COLS = ["file_path", "target_class", "label", "fold", "dataset"]


def copy_and_remap(df: pd.DataFrame, dest_dir: str) -> pd.DataFrame:
    """Copy each .npy to dest_dir, return df with updated file_path."""
    new_paths = []
    for src in tqdm(df["file_path"], desc="Copying", unit="file"):
        src = src.replace("\\", "/")
        filename = os.path.basename(src)
        dst = os.path.join(dest_dir, filename).replace("\\", "/")

        # Handle name collision: prefix with subdir name
        if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
            parent = os.path.basename(os.path.dirname(src))
            filename = f"{parent}_{filename}"
            dst = os.path.join(dest_dir, filename).replace("\\", "/")

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        new_paths.append(dst)

    df = df.copy()
    df["file_path"] = new_paths
    return df


def main():
    print("=" * 60)
    print("Merging & consolidating datasets")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load original ─────────────────────────────────────────────
    orig_df = pd.read_csv(ORIG_CSV)
    if "feature_path" in orig_df.columns and "file_path" not in orig_df.columns:
        orig_df = orig_df.rename(columns={"feature_path": "file_path"})
    orig_df["file_path"] = orig_df["file_path"].str.replace("\\", "/", regex=False)
    orig_df = orig_df[COLS]
    print(f"\nOriginal  : {len(orig_df):>6} samples")
    orig_df = copy_and_remap(orig_df, DATA_DIR)

    # ── Load augmented ────────────────────────────────────────────
    aug_df = pd.read_csv(AUG_CSV)
    aug_df["file_path"] = aug_df["file_path"].str.replace("\\", "/", regex=False)
    aug_df = aug_df[COLS]
    print(f"Augmented : {len(aug_df):>6} samples")
    aug_df = copy_and_remap(aug_df, DATA_DIR)

    # ── Merge & shuffle ───────────────────────────────────────────
    final_df = (
        pd.concat([orig_df, aug_df], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal     : {len(final_df):>6} samples")
    print(f".npy dir  →  {DATA_DIR}")
    print(f"CSV       →  {OUTPUT_CSV}")
    print("\nClass distribution:")
    print(final_df["target_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
