from pathlib import Path
import shutil
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("datasets/datasets_full_size")
OUTPUT_ROOT = Path("datasets/datasets_quarter_size")

DATASETS_TO_RESIZE = ["O-HAZE", "NH-HAZE"]
DATASETS_TO_COPY = ["RESIDE"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SCALE_FACTOR = 0.25


# ============================================================
# COPY AND RESIZE DATASETS
# ============================================================

def process_dataset(dataset_name, resize=True):

    source_dir = SOURCE_ROOT / dataset_name
    output_dir = OUTPUT_ROOT / dataset_name

    if not source_dir.exists():
        print(f"Dataset not found: {source_dir}")
        return

    image_count = 0

    for source_path in source_dir.rglob("*"):

        if not source_path.is_file():
            continue

        relative_path = source_path.relative_to(source_dir)
        destination_path = output_dir / relative_path

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Copy non-image files without modification
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            shutil.copy2(source_path, destination_path)
            continue

        # RESIDE: copy images unchanged
        if not resize:
            shutil.copy2(source_path, destination_path)

        else:
            image = cv2.imread(
                str(source_path),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                print(f"Could not read: {source_path}")
                continue

            height, width = image.shape[:2]

            new_width = max(1, round(width * SCALE_FACTOR))
            new_height = max(1, round(height * SCALE_FACTOR))

            resized = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

            success = cv2.imwrite(
                str(destination_path),
                resized
            )

            if not success:
                print(f"Could not save: {destination_path}")
                continue

        image_count += 1

        if image_count % 50 == 0:
            print(f"{dataset_name}: processed {image_count} images")

    print(f"Finished {dataset_name}: {image_count} images")


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS_TO_RESIZE:
        process_dataset(dataset, resize=True)

    for dataset in DATASETS_TO_COPY:
        process_dataset(dataset, resize=False)

    print("\nAll datasets processed.")
    print(f"Output folder: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()