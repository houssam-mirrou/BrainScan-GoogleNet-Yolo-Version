
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import hashlib
import shutil
import csv


# 1. Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data/raw/brain_mri"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROCESSED_DIR / "classification"

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_STATE = 42


# 2. Calculate image hash

def get_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# 3. Collect images and group exact duplicates

groups = defaultdict(list)
total_images = 0

for original_split in ["Train", "Val"]:
    for class_name in CLASSES:

        images_dir = RAW_DIR / original_split / class_name / "images"

        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)

        for image_path in sorted(images_dir.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in EXTENSIONS:
                continue

            image_hash = get_hash(image_path)

            groups[image_hash].append({
                "hash": image_hash,
                "class_name": class_name,
                "source": image_path,
            })

            total_images += 1


# 4. Keep one image per hash

unique_images = []

for image_hash, copies in sorted(groups.items()):

    # Detect unexpected image-level label conflicts
    labels = {copy["class_name"] for copy in copies}

    if len(labels) != 1:
        raise ValueError(
            f"Conflicting image labels for hash {image_hash}"
        )

    # Deterministic selection of one representative
    selected = sorted(
        copies,
        key=lambda item: str(item["source"])
    )[0]

    unique_images.append(selected)


print("Original images:", total_images)
print("Unique images:", len(unique_images))
print("Removed redundant copies:", total_images - len(unique_images))


# 5. Stratified splitting

labels = [item["class_name"] for item in unique_images]

train, remaining = train_test_split(
    unique_images,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=labels,
)

remaining_labels = [
    item["class_name"] for item in remaining
]

val, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=remaining_labels,
)

splits = {
    "train": train,
    "val": val,
    "test": test,
}


# 6. Verify split independence

split_hashes = {
    name: {item["hash"] for item in records}
    for name, records in splits.items()
}

assert split_hashes["train"].isdisjoint(split_hashes["val"])
assert split_hashes["train"].isdisjoint(split_hashes["test"])
assert split_hashes["val"].isdisjoint(split_hashes["test"])


# 7. Save processed images and manifest

manifest_path = PROCESSED_DIR / "split_manifest.csv"

# Prevent accidental mixing with previous outputs
if manifest_path.exists() or (
    OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir())
):
    raise FileExistsError(
        "Processed classification data already exists. "
        "Move or remove the previous generated output before rebuilding."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_records = []

for split_name, records in splits.items():

    for item in records:

        destination_dir = (
            OUTPUT_DIR / split_name / item["class_name"]
        )

        destination_dir.mkdir(parents=True, exist_ok=True)

        # Use a content-based filename to prevent collisions
        filename = item["hash"] + item["source"].suffix.lower()

        destination = destination_dir / filename

        shutil.copy2(item["source"], destination)

        manifest_records.append({
            "hash": item["hash"],
            "class_name": item["class_name"],
            "split": split_name,
            "source_image": str(
                item["source"].relative_to(PROJECT_ROOT)
            ),
            "processed_image": str(
                destination.relative_to(PROJECT_ROOT)
            ),
        })


with open(manifest_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "hash",
            "class_name",
            "split",
            "source_image",
            "processed_image",
        ],
    )

    writer.writeheader()
    writer.writerows(manifest_records)


# 8. Display final distribution

print("\nFinal classification dataset:")

for split_name, records in splits.items():

    counts = Counter(
        item["class_name"] for item in records
    )

    print(f"\n{split_name.upper()}: {len(records)} images")

    for class_name in CLASSES:
        print(f"  {class_name}: {counts[class_name]}")

print("\nManifest saved:", manifest_path)
print("Dataset preparation completed.")
