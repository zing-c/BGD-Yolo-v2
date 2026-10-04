"""CPU regression for historical mean CAM and shared corrected scatter."""

import unittest

import cv2
import numpy as np
import torch

from experiments.direct_geometry_fusion import aggregate_head_cams, normalized_cam


class MultiCamTests(unittest.TestCase):
    def test_one_head_is_exactly_single_head_formula(self):
        torch.manual_seed(0)
        activation = torch.randn(2, 5, 10, 20)
        gradient = torch.randn_like(activation)
        actual, heads = aggregate_head_cams([activation], [gradient], (160, 320))
        torch.testing.assert_close(actual, normalized_cam(activation, gradient, (160, 320)), rtol=0, atol=0)
        self.assertEqual(len(heads), 1)

    def test_three_heads_match_independent_numpy_opencv_legacy_formula(self):
        rng = np.random.default_rng(4)
        acts = [rng.normal(size=(1, 4, h, w)).astype(np.float32) for h, w in [(20, 40), (10, 20), (5, 10)]]
        grads = [rng.normal(size=a.shape).astype(np.float32) for a in acts]
        reference = []
        for act, grad in zip(acts, grads):
            cam = np.maximum((act * grad.mean(axis=(2, 3), keepdims=True)).sum(axis=1)[0], 0)
            cam -= cam.min()
            cam /= cam.max() + 1e-7
            reference.append(cv2.resize(cam, (320, 160), interpolation=cv2.INTER_LINEAR))
        expected = np.stack(reference).mean(axis=0)
        expected -= expected.min()
        expected /= expected.max() + 1e-7
        actual, heads = aggregate_head_cams([torch.from_numpy(a) for a in acts], [torch.from_numpy(g) for g in grads], (160, 320))
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=5e-6, rtol=5e-6)
        self.assertEqual(tuple(actual.shape), (1, 160, 320))
        self.assertFalse(actual.requires_grad)

    def test_zero_head_not_fabricated_and_validation(self):
        act = torch.ones(2, 3, 5, 10, requires_grad=True)
        actual, heads = aggregate_head_cams([act, act], [-torch.ones_like(act), torch.zeros_like(act)], (80, 160))
        self.assertEqual(int(actual.count_nonzero()), 0)
        self.assertTrue(all(int(c.count_nonzero()) == 0 for c in heads))
        with self.assertRaises(ValueError):
            aggregate_head_cams([], [], (80, 160))
        with self.assertRaises(FloatingPointError):
            aggregate_head_cams([act], [torch.full_like(act, float('nan'))], (80, 160))


if __name__ == '__main__':
    unittest.main(verbosity=2)
