
from pathlib import Path
import json
import math
import shutil
import pandas as pd

# 1. Configuration
ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "detection"

MANIFEST = PROCESSED / "split_manifest.csv"
AUDIT = PROCESSED / "detection_annotation_audit.csv"

CLASS_TO_ORIGINAL_ID = {
    "Glioma": 0,
    "Meningioma": 1,
    "Pituitary": 3,
}

CLASS_REMAP = {
    0: 0,
    1: 1,
    3: 2,
}

ACCEPTED = {"valid_tumor", "valid_negative"}


# 2. Load and combine reports
manifest = pd.read_csv(
    MANIFEST, dtype={"hash": str}
)
audit = pd.read_csv(
    AUDIT, dtype={"hash": str}
)

data = manifest.merge(
    audit[["hash", "split", "status", "reason"]],
    on=["hash", "split"],
    how="left",
    validate="one_to_one",
)

assert len(data) == len(manifest)
assert data["status"].notna().all()

ready = data[data["status"].isin(ACCEPTED)].copy()
excluded = data[~data["status"].isin(ACCEPTED)].copy()

print("Accepted:", len(ready))
print("Excluded:", len(excluded))


# 3. Prevent accidental output overwrite
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"Detection output already exists: {OUTPUT}"
    )


# 4. Convert validated tumor annotations
def convert_tumor_labels(label_path, expected_id):
    lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    converted = []

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO annotation: {label_path}"
            )

        old_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if old_id != expected_id:
            raise ValueError(
                f"Unexpected class {old_id}: {label_path}"
            )

        x, y, w, h = coords

        if not all(map(math.isfinite, coords)):
            raise ValueError("Non-finite coordinates")

        if not (
            0 <= x <= 1 and 0 <= y <= 1
            and 0 < w <= 1 and 0 < h <= 1
        ):
            raise ValueError("Invalid coordinates")

        tolerance = 1e-5

        if (
            x - w / 2 < -tolerance
            or x + w / 2 > 1 + tolerance
            or y - h / 2 < -tolerance
            or y + h / 2 > 1 + tolerance
        ):
            raise ValueError("Box outside image")

        new_id = CLASS_REMAP[old_id]

        converted.append(
            " ".join([str(new_id)] + parts[1:])
        )

    if not converted:
        raise ValueError(
            f"Empty positive annotation: {label_path}"
        )

    return converted


# 5. Prepare output folders
for split in ("train", "val", "test"):
    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )


# 6. Copy images and write annotations
for row in ready.itertuples(index=False):

    split = row.split
    class_name = row.class_name

    source_image = ROOT / row.source_image
    processed_image = ROOT / row.processed_image

    if not processed_image.is_file():
        raise FileNotFoundError(processed_image)

    source_label = (
        source_image.parent.parent
        / "labels"
        / f"{source_image.stem}.txt"
    )

    destination_image = (
        OUTPUT / "images" / split
        / processed_image.name
    )

    destination_label = (
        OUTPUT / "labels" / split
        / f"{processed_image.stem}.txt"
    )

    if not source_label.is_file():
        raise FileNotFoundError(source_label)

    if row.status == "valid_negative":
        if class_name != "No Tumor":
            raise ValueError(
                "Unexpected negative sample class"
            )

        # No tumor object to detect
        new_labels = []

    else:
        expected_id = CLASS_TO_ORIGINAL_ID[class_name]

        new_labels = convert_tumor_labels(
            source_label,
            expected_id,
        )

    shutil.copy2(
        processed_image,
        destination_image,
    )

    destination_label.write_text(
        "\n".join(new_labels) + (
            "\n" if new_labels else ""
        ),
        encoding="utf-8",
    )


# 7. Save tracking reports
ready.to_csv(
    OUTPUT / "detection_manifest.csv",
    index=False,
)

excluded.to_csv(
    OUTPUT / "excluded_for_review.csv",
    index=False,
)


# 8. Generate data.yaml
yaml_content = f"""path: {json.dumps(str(OUTPUT))}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: Glioma
  1: Meningioma
  2: Pituitary
"""

(OUTPUT / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8",
)


# 9. Final summary
print("\n===== YOLO DATASET PREPARATION =====")

for split in ("train", "val", "test"):

    images = list(
        (OUTPUT / "images" / split).iterdir()
    )

    labels = list(
        (OUTPUT / "labels" / split).glob("*.txt")
    )

    negative_count = sum(
        not p.read_text().strip()
        for p in labels
    )

    assert len(images) == len(labels)

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Negative samples:", negative_count)

print("\nExcluded:", len(excluded))
print("Dataset:", OUTPUT)
print("Configuration:", OUTPUT / "data.yaml")
print("\nPreparation completed.")
