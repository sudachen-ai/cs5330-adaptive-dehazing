"""Regression tests for pairing and held-out scene isolation."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from prepare_data import find_pairs, split_dataset
from evaluate_dehazing import calculate_metrics


class EvaluationTests(unittest.TestCase):
    def test_multiple_haze_variants_stay_in_same_split(self):
        frame = pd.DataFrame({'scene_id': [str(i) for i in range(10) for _ in range(3)]})
        validation, test = split_dataset(frame)
        self.assertFalse(set(validation.scene_id) & set(test.scene_id))
        self.assertEqual(len(validation) + len(test), len(frame))
        pd.testing.assert_frame_equal(test, split_dataset(frame)[1])

    def test_pairing_allows_different_extensions_and_rejects_missing_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hazy = root / 'example/hazy/outdoor/001_hazy_outdoor.jpg'
            gt = root / 'example/GT/outdoor/001_GT_outdoor.png'
            for path in (hazy, gt):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            pairs = find_pairs(root, 'example')
            self.assertEqual(pairs.iloc[0].gt_path, gt.relative_to(root).as_posix())
            gt.unlink()
            with self.assertRaises(ValueError):
                find_pairs(root, 'example')

    def test_metrics_use_unit_range_and_reject_shape_mismatch(self):
        clear = np.full((8, 8, 3), 0.5, np.float32)
        predicted = clear + 0.1
        psnr, ssim = calculate_metrics(clear, predicted)
        self.assertAlmostEqual(psnr, 20, places=4)
        self.assertGreater(ssim, 0.9)
        with self.assertRaises(ValueError):
            calculate_metrics(clear, predicted[:7])
