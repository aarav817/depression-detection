"""Summarize facial action-unit intensities into participant-level features.

Run: python src/dataset.py
Only frames with success == 1 contribute. Statistics ignore missing/nonfinite
AU values; population standard deviation (ddof=0) is used. No imputation or
scaling is fitted here. Those steps must be fitted on training participants.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
AU_COLUMNS = [
    f"AU{number:02d}_r"
    for number in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
]
STATISTICS = ["mean", "std", "median", "iqr"]
FEATURE_COLUMNS = [f"{au}_{stat}" for au in AU_COLUMNS for stat in STATISTICS]


def summarize_visual(path):
    """Return a feature dictionary (or None) and a data-quality record."""
    quality = {
        "status": "excluded", "reason": "", "total_frames": 0,
        "successful_frames": 0, "discarded_frames": 0,
        "invalid_au_values": 0, "out_of_range_au_values": 0,
        "missing_features": 0,
    }
    wanted = {"success", *AU_COLUMNS}
    try:
        frame = pd.read_csv(path, usecols=lambda name: name.strip() in wanted)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        quality["reason"] = f"Cannot read visual CSV: {error}"
        return None, quality

    frame.columns = frame.columns.str.strip()
    quality["total_frames"] = len(frame)
    missing = sorted(wanted - set(frame.columns))
    if missing:
        quality["reason"] = "Missing columns: " + ", ".join(missing)
        return None, quality

    successful = pd.to_numeric(frame["success"], errors="coerce").eq(1)
    quality["successful_frames"] = int(successful.sum())
    quality["discarded_frames"] = int((~successful).sum())
    if not successful.any():
        quality["reason"] = "No successfully tracked frames"
        return None, quality

    values = frame.loc[successful, AU_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    quality["invalid_au_values"] = int(values.isna().sum().sum())
    # OpenFace AU intensities have a documented 0–5 range. Treat values outside
    # that range as missing and report them, rather than clipping them silently.
    out_of_range = values.lt(0) | values.gt(5)
    quality["out_of_range_au_values"] = int(out_of_range.sum().sum())
    values = values.mask(out_of_range)
    if not values.notna().any().any():
        quality["reason"] = "No usable AU measurements in successfully tracked frames"
        return None, quality

    stats = {
        "mean": values.mean(),
        "std": values.std(ddof=0),
        "median": values.median(),
        "iqr": values.quantile(0.75) - values.quantile(0.25),
    }
    features = {f"{au}_{stat}": float(stats[stat][au]) for au in AU_COLUMNS for stat in STATISTICS}
    quality["missing_features"] = sum(pd.isna(value) for value in features.values())
    warnings = []
    if quality["invalid_au_values"]:
        warnings.append("Missing, nonnumeric, or infinite AU measurements ignored")
    if quality["out_of_range_au_values"]:
        warnings.append("AU measurements outside 0–5 ignored")
    if quality["missing_features"]:
        warnings.append("Some AUs have no usable measurements; their features remain missing")
    if quality["successful_frames"] == 1:
        warnings.append("Only one successfully tracked frame; variability is not reliable")
    quality["status"] = "warning" if warnings else "ok"
    quality["reason"] = "; ".join(warnings)
    return features, quality


def main():
    manifest = pd.read_csv(PROCESSED / "manifest.csv")
    if manifest["participant_key"].isna().any() or not manifest["participant_key"].is_unique:
        raise ValueError("Manifest must contain one unique, nonmissing key per participant.")

    feature_rows, quality_rows = [], []
    for index, row in enumerate(manifest.itertuples(index=False), 1):
        features, quality = summarize_visual(PROJECT / row.visual_path)
        quality_rows.append({"participant_key": row.participant_key, **quality})
        if features is not None:
            feature_rows.append({"participant_key": row.participant_key, **features})
        if index % 100 == 0 or index == len(manifest):
            print(f"Processed {index}/{len(manifest)} participants", flush=True)

    features = pd.DataFrame(feature_rows, columns=["participant_key", *FEATURE_COLUMNS])
    quality = pd.DataFrame(quality_rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED / "features.csv", index=False)
    quality.to_csv(PROCESSED / "feature_quality.csv", index=False)
    metadata = {
        "source_manifest": "data/processed/manifest.csv",
        "frame_filter": "success == 1",
        "au_columns": AU_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "statistics": STATISTICS,
        "standard_deviation_ddof": 0,
        "quantile_interpolation": "linear",
        "invalid_values": "Missing, nonnumeric, nonfinite, or outside [0, 5]; ignored per AU",
        "imputation": "None; fit later on training data only",
        "scaling": "None; fit later on training data only",
        "participant_count": len(manifest),
        "feature_row_count": len(features),
        "quality_status_counts": {key: int(value) for key, value in quality.status.value_counts().items()},
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (PROCESSED / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved {len(features)} participant rows with {len(FEATURE_COLUMNS)} features each.")
    print(quality.status.value_counts().to_string())
    print("Data-quality details: data/processed/feature_quality.csv")


if __name__ == "__main__":
    main()
