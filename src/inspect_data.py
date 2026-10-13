#for future reference: this file takes the raw data, creates a dataframe with every file
#in both the hospital and school datasets, uses a bunch of assert manifest statements to 
#catch errors, then prints the first five rows of facial measurements based on the AU##_r columns

from pathlib import Path

import pandas as pd

# Locate the data relative to this script.
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "raw" / "AVDD-TCMI-dataset"
OUTPUT = PROJECT / "data" / "processed"

tables = []

for subset in ["HDS", "SDS"]:
    # These label files do not contain a header row.
    labels = pd.read_csv(
        DATA / f"{subset}_labels.csv",
        header=None,
        names=["participant_id", "label", "gender"],
        dtype={"participant_id": str},
    )

    labels["subset"] = subset

    # HDS participant 0 and SDS participant 0 need different keys.
    labels["participant_key"] = (
        subset + "_" + labels["participant_id"]
    )

    labels["visual_path"] = labels["participant_id"].apply(
        lambda participant_id: (
            f"data/raw/AVDD-TCMI-dataset/{subset}/"
            f"{participant_id}/{participant_id}_visual.csv"
        )
    )

    tables.append(labels)

manifest = pd.concat(tables, ignore_index=True)

# Catch problems before training.
assert manifest["participant_key"].is_unique, "Duplicate participants!"
assert manifest["label"].isin([0, 1]).all(), "Unexpected labels!"

missing = [
    path
    for path in manifest["visual_path"]
    if not (PROJECT / path).is_file()
]
assert not missing, f"Missing visual files: {missing[:5]}"

OUTPUT.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT / "manifest.csv", index=False)

print("\nParticipants by subset and label:")
print(pd.crosstab(manifest["subset"], manifest["label"]))
print(f"\nTotal participants: {len(manifest)}")

# Inspect one participant's facial measurements.
sample_path = PROJECT / manifest.iloc[0]["visual_path"]
visual = pd.read_csv(sample_path)

# The actual CSV headers contain leading spaces.
visual.columns = visual.columns.str.strip()

au_columns = [
    column for column in visual.columns
    if column.startswith("AU") and column.endswith("_r")
]

print(f"\nExample file: {sample_path.name}")
print(f"Rows: {len(visual)}")
print(f"Columns: {len(visual.columns)}")
print(f"Action-unit intensity columns: {au_columns}")
print(f"Missing cells: {visual.isna().sum().sum()}")

if au_columns:
    print("\nFirst five rows of facial measurements:")
    print(visual[au_columns].head())

print("\nSaved: data/processed/manifest.csv")