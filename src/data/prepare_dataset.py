from pathlib import Path
from collections import Counter
import csv
import json
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/raw/nih_malaria"
IMAGE_ROOT = DATA_ROOT / "cell_images"
OUTPUT_ROOT = PROJECT_ROOT / "data/processed"

MAPPING_FILES = {
    "Parasitized": DATA_ROOT / "patientid_cellmapping_parasitized.csv",
    "Uninfected": DATA_ROOT / "patientid_cellmapping_uninfected.csv",
}

LABELS = {"Uninfected": 0, "Parasitized": 1}
records = []
all_paths = set()

for class_name, mapping_path in MAPPING_FILES.items():
    image_dir = IMAGE_ROOT / class_name
    actual_files = {p.name for p in image_dir.glob("*.png")}

    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)

    patient_to_files = {}

    for line in mapping_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue

        patient_id = line.split(",", 1)[0].strip()
        filenames = re.findall(r"[A-Za-z0-9_]+\.png", line)

        if not patient_id or not filenames:
            continue

        patient_to_files.setdefault(patient_id, []).extend(filenames)

    mapped_files = [
        filename
        for filenames in patient_to_files.values()
        for filename in filenames
    ]

    duplicates = [
        name for name, count in Counter(mapped_files).items()
        if count > 1
    ]

    missing = set(mapped_files) - actual_files
    unmapped = actual_files - set(mapped_files)

    if duplicates or missing or unmapped:
        raise RuntimeError(
            f"{class_name} mapping validation failed:\n"
            f"Duplicates: {len(duplicates)}\n"
            f"Missing files: {len(missing)}\n"
            f"Unmapped files: {len(unmapped)}"
        )

    for patient_id, filenames in patient_to_files.items():
        for filename in filenames:
            relative_path = (
                Path("data/raw/nih_malaria/cell_images")
                / class_name
                / filename
            )
            full_path = PROJECT_ROOT / relative_path

            if full_path in all_paths:
                raise RuntimeError(f"Image mapped more than once: {full_path}")

            all_paths.add(full_path)

            records.append({
                "image_path": relative_path.as_posix(),
                "label": LABELS[class_name],
                "class_name": class_name,
                "patient_id": patient_id,
            })

    print(
        f"{class_name}: {len(actual_files)} images, "
        f"{len(patient_to_files)} patients — validated"
    )

# One patient ID is one indivisible group across both classes.
y = np.array([record["label"] for record in records])
groups = np.array([record["patient_id"] for record in records])
X = np.zeros((len(records), 1))

# First reserve approximately 20% of patients/images for testing.
outer_splitter = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)
train_val_idx, test_idx = next(outer_splitter.split(X, y, groups))

# Split the remaining data into training and validation.
inner_splitter = StratifiedGroupKFold(
    n_splits=4, shuffle=True, random_state=42
)
local_train_idx, local_val_idx = next(
    inner_splitter.split(
        X[train_val_idx],
        y[train_val_idx],
        groups[train_val_idx],
    )
)

train_idx = train_val_idx[local_train_idx]
val_idx = train_val_idx[local_val_idx]

for index in train_idx:
    records[index]["split"] = "train"

for index in val_idx:
    records[index]["split"] = "validation"

for index in test_idx:
    records[index]["split"] = "test"

# Verify that no patient appears in more than one split.
patients_by_split = {
    split: {
        records[i]["patient_id"] for i in indices
    }
    for split, indices in {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }.items()
}

assert patients_by_split["train"].isdisjoint(patients_by_split["validation"])
assert patients_by_split["train"].isdisjoint(patients_by_split["test"])
assert patients_by_split["validation"].isdisjoint(patients_by_split["test"])

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT_ROOT / "splits.csv"
with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path", "label", "class_name",
            "patient_id", "split",
        ],
    )
    writer.writeheader()
    writer.writerows(records)

summary = {
    "total_images": len(records),
    "total_patients": len(set(groups)),
    "splits": {},
}

for split in ("train", "validation", "test"):
    split_records = [r for r in records if r["split"] == split]
    summary["splits"][split] = {
        "images": len(split_records),
        "patients": len({
            r["patient_id"] for r in split_records
        }),
        "uninfected": sum(r["label"] == 0 for r in split_records),
        "parasitized": sum(r["label"] == 1 for r in split_records),
    }

summary_path = OUTPUT_ROOT / "split_summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\nPATIENT-INDEPENDENT SPLIT SUMMARY")
print("=" * 48)

for split, values in summary["splits"].items():
    print(f"\n{split.upper()}")
    for key, value in values.items():
        print(f"  {key:12}: {value}")

print("\nPatient overlap check: PASSED")
print(f"Saved: {csv_path}")
print(f"Saved: {summary_path}")
