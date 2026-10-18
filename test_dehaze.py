"""Numerical and blending contract tests; run with python -m unittest -v."""
import unittest
import numpy as np
from dehaze import process, guided_filter


class PipelineTests(unittest.TestCase):
    def test_constant_black_white_and_tiny_images(self):
        for shape in [(1, 1, 3), (7, 11, 3), (32, 40, 3)]:
            for value in (0.0, 0.5, 1.0):
                image = np.full(shape, value, np.float32)
                results = process(image)
                for name, data in results.items():
                    self.assertTrue(np.isfinite(data).all(), name)
                    self.assertGreaterEqual(float(data.min()), 0, name)
                    self.assertLessEqual(float(data.max()), 1, name)
                np.testing.assert_allclose(results["adaptive"], image, atol=1e-5)

    def test_blending_stays_between_source_pixels(self):
        image = np.random.default_rng(42).random((43, 57, 3), dtype=np.float32)
        result = process(image, alpha=0)
        np.testing.assert_array_equal(result["fixed"], image)
        low = np.minimum(image, result["dcp"])
        high = np.maximum(image, result["dcp"])
        self.assertTrue((result["adaptive"] >= low - 1e-6).all())
        self.assertTrue((result["adaptive"] <= high + 1e-6).all())
        endpoint = process(image, alpha=1)
        np.testing.assert_array_equal(endpoint["fixed"], endpoint["dcp"])

    def test_guided_filter_preserves_constant_source(self):
        guide = np.random.default_rng(7).random((20, 30), dtype=np.float32)
        source = np.full_like(guide, 0.4)
        np.testing.assert_allclose(guided_filter(guide, source), source, atol=1e-5)

    def test_invalid_parameters_and_nonfinite_input(self):
        image = np.zeros((3, 4, 3), np.float32)
        for kwargs in ({"patch": 2}, {"eps": 0}, {"t_min": 0},
                       {"alpha": 2}, {"omega": -1}, {"radius": -1}):
            with self.assertRaises(ValueError):
                process(image, **kwargs)
        image[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            process(image)


if __name__ == "__main__":
    unittest.main()
