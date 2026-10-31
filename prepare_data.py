from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ROOT = Path("datasets/datasets_full_size")
OUTPUT_ROOT = Path("splits")

TEST_SIZE = 0.20
RANDOM_STATE = 42

DATASETS = ["NH-HAZE", "O-HAZE", "RESIDE"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def find_pairs(dataset):
    dataset_dir = DATASET_ROOT / dataset
    hazy_dir = dataset_dir / "hazy"
    gt_dir = dataset_dir / "GT"

    pairs = []

    for hazy_path in sorted(hazy_dir.rglob("*")):
        if not hazy_path.is_file():
            continue

        if hazy_path.suffix.lower() not in EXTENSIONS:
            continue

        relative_path = hazy_path.relative_to(hazy_dir)

        gt_filename = hazy_path.name.replace(
            "_hazy_", "_GT_"
        )

        gt_path = (
            gt_dir
            / relative_path.parent
            / gt_filename
        )

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Missing ground truth for {hazy_path}\n"
                f"Expected: {gt_path}"
            )

        # Assumes the number before the first underscore
        # identifies the original scene.
        scene_id = hazy_path.stem.split("_")[0]

        pairs.append({
            "dataset": dataset,
            "scene_id": scene_id,
            "hazy_path": str(hazy_path.resolve()),
            "gt_path": str(gt_path.resolve())
        })

    if not pairs:
        raise ValueError(f"No pairs found for {dataset}")

    return pd.DataFrame(pairs)


def split_dataset(df):
    # Split scenes, not individual images.
    scene_ids = sorted(df["scene_id"].unique())

    train_ids, test_ids = train_test_split(
        scene_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_df = df[df["scene_id"].isin(train_ids)].copy()
    test_df = df[df["scene_id"].isin(test_ids)].copy()

    return train_df, test_df


def save_split(name, datasets, splits):
    train_df = pd.concat(
        [splits[d][0] for d in datasets],
        ignore_index=True
    )

    test_df = pd.concat(
        [splits[d][1] for d in datasets],
        ignore_index=True
    )

    folder = OUTPUT_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(folder / "train.csv", index=False)
    test_df.to_csv(folder / "test.csv", index=False)

    print(f"\n{name}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")


def generate_partition():
    splits = {}

    for dataset in DATASETS:
        df = find_pairs(dataset)

        train_df, test_df = split_dataset(df)
        splits[dataset] = (train_df, test_df)

        print(
            f"{dataset}: {len(train_df)} training pairs, "
            f"{len(test_df)} testing pairs"
        )

    return splits


def generate_real_haze_split():
    splits = generate_partition()
    return save_split("real_haze", ["NH-HAZE", "O-HAZE"], splits)


def generate_all_datasets_split():
    splits = generate_partition()
    return save_split("all_datasets", ["NH-HAZE", "O-HAZE", "RESIDE"], splits)


generate_real_haze_split()
generate_all_datasets_split()
