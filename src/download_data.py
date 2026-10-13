"""Download AVDD-TCMI and extract its visual-feature and label CSVs."""

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Mar-rill/AVDD-TCMI-dataset"
ARCHIVE_NAME = "AVDD-TCMI-dataset.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_csvs(archive_path, output_dir):
    """Extract relevant CSVs, preserving their paths inside the archive."""
    output_dir = Path(output_dir).resolve()
    with ZipFile(archive_path) as archive:
        selected = []
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            name = path.name.lower()
            if member.is_dir() or "__MACOSX" in path.parts or name.startswith("._"):
                continue
            if name.endswith(".csv") and ("visual" in name or "label" in name):
                destination = (output_dir / member.filename).resolve()
                if not destination.is_relative_to(output_dir):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                selected.append((member, destination))

        if not selected:
            raise ValueError("No visual or label CSVs found. Inspect the archive layout.")

        required = sum(member.file_size for member, _ in selected)
        output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_dir).free < required:
            raise OSError(f"Extraction needs approximately {required / 1e9:.2f} GB free.")

        print(f"Extracting {len(selected)} CSVs ({required / 1e9:.2f} GB)...", flush=True)
        for index, (member, destination) in enumerate(selected, 1):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            try:
                with archive.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            if index % 100 == 0 or index == len(selected):
                print(f"Extracted {index}/{len(selected)}", flush=True)

        return [str(destination.relative_to(output_dir)) for _, destination in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-only", action="store_true", help="Inspect remote files without downloading.")
    parser.add_argument("--archive", type=Path, help="Extract a ZIP already downloaded, without network access.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    revision = None
    archive_path = args.archive
    if archive_path is None:
        info = HfApi().dataset_info(REPO_ID, files_metadata=True)
        revision = info.sha
        for item in info.siblings:
            print(f"{item.rfilename}: {(item.size or 0) / 1e9:.3f} GB", flush=True)
        if args.list_only:
            return
        print("Downloading the full ZIP; CSVs are not hosted individually.", flush=True)
        print("If interrupted, rerun this command to reuse the download cache.", flush=True)
        archive_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ARCHIVE_NAME,
            revision=revision,
            local_dir=PROJECT_ROOT / "data" / "downloads",
        ))

    extracted = extract_csvs(archive_path, args.output_dir)
    metadata = {
        "repo_id": REPO_ID,
        "revision": revision,
        "archive": str(archive_path.resolve()),
        "extracted_files": extracted,
    }
    (args.output_dir / "download_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Done. CSV files are in {args.output_dir.resolve()}", flush=True)
    print("Label files:", flush=True)
    for filename in extracted:
        if "label" in PurePosixPath(filename).name.lower():
            print(f"  {args.output_dir / filename}", flush=True)


if __name__ == "__main__":
    main()
