"""Create fixed participant splits stratified by subset and depression label.

Run from the project root: python src/split_data.py
All frames and derived features must inherit their participant's assignment.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
SEED = 42


def create_splits(manifest):
    required = {"participant_key", "subset", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Participant keys, subsets, and labels cannot be missing.")
    if not manifest["participant_key"].is_unique:
        raise ValueError("Manifest must have exactly one row per participant.")
    if not manifest["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    # Sorting makes assignments independent of the manifest's row order.
    participants = manifest.sort_values("participant_key").reset_index(drop=True).copy()
    participants["stratum"] = (
        participants["subset"].astype(str) + "_" + participants["label"].astype(str)
    )
    train, held_out = train_test_split(
        participants,
        test_size=0.30,
        stratify=participants["stratum"],
        random_state=SEED,
    )
    validation, test = train_test_split(
        held_out,
        test_size=0.50,
        stratify=held_out["stratum"],
        random_state=SEED,
    )
    splits = pd.concat(
        [group[["participant_key"]].assign(split=name)
         for name, group in [("train", train), ("validation", validation), ("test", test)]],
        ignore_index=True,
    ).sort_values("participant_key").reset_index(drop=True)

    if not splits["participant_key"].is_unique:
        raise ValueError("A participant was assigned to multiple splits.")
    if set(splits["participant_key"]) != set(manifest["participant_key"]):
        raise ValueError("Split assignments do not cover the manifest exactly.")
    return splits


def main():
    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    splits = create_splits(manifest)
    output = PROCESSED / "splits.csv"
    if output.exists():
        previous = pd.read_csv(output).sort_values("participant_key").reset_index(drop=True)
        if not previous.equals(splits):
            raise ValueError("Existing splits differ. Preserve the original evaluation groups; investigate before replacing them.")
        print("Existing splits match; kept the original file.")
    else:
        splits.to_csv(output, index=False)
        print(f"Saved: {output}")

    report = manifest.merge(splits, on="participant_key", validate="one_to_one")
    print(f"\nRandom seed: {SEED}")
    print("\nParticipant counts:")
    print(report["split"].value_counts().reindex(["train", "validation", "test"]))
    print("\nCounts by subset and label:")
    print(pd.crosstab([report["subset"], report["label"]], report["split"])
          .reindex(columns=["train", "validation", "test"]))


if __name__ == "__main__":
    main()
