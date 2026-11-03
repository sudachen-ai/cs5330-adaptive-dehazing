"""Create reproducible, scene-disjoint validation/test manifests with portable paths."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def find_pairs(root, dataset):
    rows = []
    base = root / dataset
    for hazy in sorted((base / 'hazy').rglob('*')):
        if hazy.suffix.lower() not in EXTENSIONS or not hazy.is_file():
            continue
        relative = hazy.relative_to(base / 'hazy')
        expected = hazy.stem.lower().replace('_hazy_', '_gt_', 1)
        matches = [p for p in (base / 'GT' / relative.parent).glob('*')
                   if p.suffix.lower() in EXTENSIONS and p.stem.lower() == expected]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one GT match for {hazy}; got {len(matches)}')
        category = relative.parent.as_posix()
        rows.append(dict(dataset=dataset, category=category,
                         scene_id=category + '/' + hazy.stem.split('_')[0],
                         hazy_path=hazy.relative_to(root).as_posix(),
                         gt_path=matches[0].relative_to(root).as_posix()))
    if not rows:
        raise ValueError(f'No pairs found for {dataset}')
    return pd.DataFrame(rows)


def split_dataset(frame, test_size=0.2, seed=42):
    scenes = sorted(frame.scene_id.unique())
    validation, test = train_test_split(scenes, test_size=test_size, random_state=seed)
    return (frame[frame.scene_id.isin(validation)].copy(),
            frame[frame.scene_id.isin(test)].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('datasets/datasets_quarter_size'))
    parser.add_argument('--datasets', nargs='+', default=['NH-HAZE', 'O-HAZE'])
    parser.add_argument('--output', type=Path, default=Path('splits/real_haze'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    partitions = [split_dataset(find_pairs(args.data_root, name), args.test_size, args.seed)
                  for name in args.datasets]
    args.output.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(('validation', 'test')):
        frame = pd.concat([pair[i] for pair in partitions], ignore_index=True)
        frame.to_csv(args.output / f'{name}.csv', index=False)
        print(f'{name}: {len(frame)} pairs')


if __name__ == '__main__':
    main()
