"""CPU geometry regressions; GPU end-to-end tests live in audit_direct_geometry.py."""

import os
import random
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import patch

os.environ.setdefault('WANDB_MODE', 'disabled')

import cv2
import numpy as np
import torch

from ultralytics.yolo.data.augment import LetterBox, Mosaic, RandomFlip, RandomHSV, RandomPerspective
from ultralytics.yolo.data.direct_geometry import (
    KEY, initial_sources, update_sources, native_crop, transform_rect, apply_hsv,
)
from ultralytics.yolo.utils.instance import Instances
from experiments.direct_geometry_fusion import (
    normalized_cam, centers_for_sources, scatter_crops, scatter_fixed_grid, scatter_input_support,
    build_crops, replace_logits, resize_detail, update_cam_counts,
)


def label(image, path='synthetic.png', original_hw=None):
    height, width = image.shape[:2]
    return {'img': image.copy(), 'im_file': path, 'ori_shape': original_hw or (height, width),
            'resized_shape': (height, width), 'ratio_pad': (1., 1.),
            'instances': Instances(np.array([[.5, .5, .5, .5]], dtype=np.float32),
                                   bbox_format='xywh', normalized=True),
            'cls': np.array([[0]], dtype=np.float32),
            KEY: initial_sources(path, original_hw or (height, width), (height, width))}


class GeometryTests(unittest.TestCase):
    def test_non_square_cam_and_marker(self):
        activation = torch.zeros(1, 1, 3, 7)
        activation[0, 0, 1, 5] = 1
        cam = normalized_cam(activation, torch.ones_like(activation), (48, 112))
        self.assertEqual(tuple(cam.shape), (1, 48, 112))
        y, x = divmod(int(cam.argmax()), 112)
        self.assertTrue(16 <= y < 32 and 80 <= x < 96)

    def test_all_ones_window_containment_and_rectangular_gains(self):
        cam = torch.zeros(96, 160)
        cam[55:70, 100:130] = 10  # tempt legacy search outside candidate
        source = initial_sources('x', (192, 640), (96, 160))[0]
        boxes = torch.tensor([[40., 20., 104., 60.]])
        centers, owners, windows = centers_for_sources(cam, boxes, [source])
        self.assertEqual(windows, [(32, 16)])
        x, y = centers[0]
        self.assertTrue(40 <= x - 8 and x + 8 <= 104)
        self.assertTrue(20 <= y - 16 and y + 16 <= 60)
        self.assertIs(owners[0], source)

    def test_zero_cam_small_box_and_padding(self):
        sources = initial_sources('x', (128, 128), (64, 64))
        cam = torch.zeros(80, 96)
        centers, owners, _ = centers_for_sources(cam, torch.tensor([
            [0., 0., 50., 50.], [5., 7., 8., 9.], [75., 65., 90., 75.]]), sources)
        np.testing.assert_equal(centers[0], [25, 25])
        np.testing.assert_equal(centers[1], [6.5, 8])
        self.assertIsNone(centers[2])
        self.assertIsNone(owners[2])

    def test_window_larger_than_canvas(self):
        source = initial_sources('x', (16, 16), (32, 32))[0]
        centers, _, windows = centers_for_sources(torch.ones(32, 32), torch.tensor([[0., 0., 32., 32.]]), [source])
        np.testing.assert_equal(centers[0], [16, 16])
        self.assertEqual(windows, [(128, 128)])

    def test_true_border_crop_and_small_source(self):
        image = np.arange(100 * 150 * 3, dtype=np.uint8).reshape(100, 150, 3)
        crop, rect = native_crop(image, (149, 99))
        np.testing.assert_equal(rect, [86, 36, 150, 100])
        np.testing.assert_array_equal(crop, image[36:100, 86:150])
        crop, rect = native_crop(np.ones((20, 30, 3), dtype=np.uint8), (29, 19))
        self.assertEqual(crop.shape, (64, 64, 3))
        np.testing.assert_equal(rect, [-17, -22, 47, 42])

    def test_letterbox_actual_integer_padding(self):
        image = np.zeros((101, 203, 3), np.uint8)
        transformed = LetterBox((192, 256))(label(image))
        source = transformed[KEY][0]
        expected_width, expected_height = 256, round(101 * 256 / 203)
        np.testing.assert_allclose(source['matrix'], [[expected_width / 203, 0, 0],
                                                    [0, expected_height / 101, 32], [0, 0, 1]])
        self.assertEqual(transformed['img'].shape[:2], (192, 256))

    def test_flip_and_hsv_match_global_pixels(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)
        transformed = RandomFlip(p=1)(label(image))
        np.testing.assert_array_equal(transformed['img'], np.fliplr(image))
        np.testing.assert_allclose(transformed[KEY][0]['matrix'], [[-1, 0, 150], [0, 1, 0], [0, 0, 1]])
        transformed = RandomHSV(.015, .7, .4)(label(image))
        np.testing.assert_array_equal(apply_hsv(image, transformed[KEY][0]['hsv']), transformed['img'])

    def test_mosaic_preserves_four_source_owners(self):
        labels = [label(np.full((64, 64, 3), i * 40, np.uint8), str(i)) for i in range(4)]
        labels[0]['mix_labels'] = labels[1:]
        with patch('random.uniform', return_value=64):
            combined = Mosaic(None, imgsz=64)._mosaic4(labels[0])
        self.assertEqual([source['path'] for source in combined[KEY]], ['0', '1', '2', '3'])
        for source, expected in zip(combined[KEY], [[0, 0, 64, 64], [64, 0, 128, 64],
                                                   [0, 64, 64, 128], [64, 64, 128, 128]]):
            np.testing.assert_equal(source['visible'], expected)
        centers, owners, _ = centers_for_sources(torch.zeros(128, 128), torch.tensor([
            [10., 10., 50., 50.], [70., 10., 110., 50.],
            [10., 70., 50., 110.], [70., 70., 110., 110.]]), combined[KEY])
        self.assertEqual([source['path'] for source in owners], ['0', '1', '2', '3'])
        for center, source, expected in zip(centers, owners, [[30, 30], [26, 30], [30, 26], [26, 26]]):
            native = np.linalg.inv(source['matrix']) @ np.r_[center, 1]
            np.testing.assert_equal(native[:2], expected)

    def test_geometry_metadata_does_not_change_augmentation(self):
        rng = np.random.default_rng(8)
        with_meta = label(rng.integers(0, 256, (80, 112, 3), dtype=np.uint8))
        without_meta = deepcopy(with_meta)
        without_meta.pop(KEY)
        transform = RandomPerspective(scale=.5, translate=.1)
        random.seed(12)
        first = transform(with_meta)
        random.seed(12)
        second = transform(without_meta)
        np.testing.assert_array_equal(first['img'], second['img'])
        np.testing.assert_array_equal(first['instances'].bboxes, second['instances'].bboxes)

    def test_unsupported_geometry_fails_closed(self):
        sample = label(np.zeros((80, 80, 3), np.uint8))
        with self.assertRaises(ValueError):
            update_sources(sample, np.array([[1., .1, 0], [0, 1, 0], [0, 0, 1]]), [0, 0, 80, 80])

    def test_sum_xy_actual_footprint_and_gradients_all_heads(self):
        records = [{'batch': 0, 'input_rect': [160., 64., 192., 96.],
                    'visible_rect': [160., 64., 192., 96.]}] * 2
        for stride in (8, 16, 32):
            values = torch.ones(2, 3, 4, 4, requires_grad=True)
            feature = scatter_crops(values, records, 2, (320 // stride, 640 // stride), (320, 640))
            x0, x1, y0, y1 = 160 // stride, 192 // stride, 64 // stride, 96 // stride
            self.assertTrue(torch.all(feature[0, :, y0:y1, x0:x1] == 2))
            self.assertEqual(int(torch.count_nonzero(feature)), 3 * (x1 - x0) * (y1 - y0))
            self.assertEqual(float(feature[1].sum()), 0)
            feature.sum().backward()
            self.assertGreater(float(values.grad.abs().sum()), 0)
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_visibility_clip_does_not_stretch_and_mean(self):
        values = torch.arange(16.).reshape(1, 1, 4, 4).requires_grad_()
        record = {'batch': 0, 'input_rect': [0., 0., 64., 64.], 'visible_rect': [32., 0., 64., 64.]}
        result = scatter_crops(values, [record], 1, (4, 4), (64, 64))
        torch.testing.assert_close(result[0, 0, :, 2:], values[0, 0, :, 2:])
        self.assertEqual(float(result[0, 0, :, :2].sum()), 0)
        summed = scatter_crops(values.repeat(2, 1, 1, 1), [record, record], 1, (4, 4), (64, 64))
        averaged = scatter_crops(values.repeat(2, 1, 1, 1), [record, record], 1, (4, 4), (64, 64), 'mean')
        torch.testing.assert_close(summed, result * 2)
        torch.testing.assert_close(averaged, result)

    def test_fixed_grid_keeps_every_detail_cell_sum_xy_and_gradients_p4_p5(self):
        record = {'batch': 0, 'input_rect': [160., 64., 192., 96.]}
        for stride in (16, 32):
            values = torch.arange(48.).reshape(1, 3, 4, 4).repeat(2, 1, 1, 1).requires_grad_()
            result, written = scatter_fixed_grid(values, [record, record], 2,
                (320 // stride, 640 // stride), (320, 640), return_records=True)
            cx, cy = round(176 / stride), round(80 / stride)
            torch.testing.assert_close(result[0, :, cy - 2:cy + 2, cx - 2:cx + 2], values[0] * 2)
            self.assertEqual(len(written), 2)
            self.assertEqual(float(result[1].sum()), 0)
            result.sum().backward()
            torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_fixed_grid_size_independent_of_source_gain(self):
        values = torch.arange(48.).reshape(1, 3, 4, 4)
        results = []
        for width in (16., 32., 64., 128.):
            record = {'batch': 0, 'input_rect': [176 - width / 2, 80 - width / 2,
                                               176 + width / 2, 80 + width / 2]}
            results.append(scatter_fixed_grid(values, [record], 1, (20, 40), (320, 640)))
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], rtol=0, atol=0)

    def test_fixed_grid_border_clips_without_shift_or_resize(self):
        value = torch.arange(16.).reshape(1, 1, 4, 4).requires_grad_()
        record = {'batch': 0, 'input_rect': [-16., -16., 16., 16.]}
        result = scatter_fixed_grid(value, [record], 1, (20, 40), (320, 640))
        torch.testing.assert_close(result[0, 0, :2, :2], value[0, 0, 2:, 2:])
        self.assertEqual(int(torch.count_nonzero(result)), 4)
        result.sum().backward()
        expected = torch.zeros_like(value)
        expected[..., 2:, 2:] = 1
        torch.testing.assert_close(value.grad, expected)

    def test_fixed_grid_actual_border_crop_center_not_requested_hotspot(self):
        value = torch.ones(1, 1, 4, 4)
        # The clamped crop projects to x=0..32,y=18..50, not its requested corner hotspot.
        record = {'batch': 0, 'input_rect': [0., 18., 32., 50.], 'center_xy': [2.5, 47.5]}
        result = scatter_fixed_grid(value, [record], 1, (10, 10), (160, 160))
        expected = torch.zeros_like(result)
        expected[0, :, :4, :3] = 1
        torch.testing.assert_close(result, expected)

    def test_fixed_grid_no_writes_keeps_original_logits(self):
        value = torch.ones(1, 3, 4, 4)
        record = {'batch': 0, 'input_rect': [-256., 32., 16., 96.]}
        feature, written = scatter_fixed_grid(value, [record], 1, (20, 40), (320, 640),
                                              return_records=True)
        self.assertEqual(written, [])
        self.assertEqual(float(feature.sum()), 0)
        raw, logits = torch.randn(1, 65, 20, 40), torch.randn(1, 1, 20, 40)
        torch.testing.assert_close(replace_logits(raw, logits, written, 64), raw, rtol=0, atol=0)

    def test_fixed_grid_mean_and_invalid_metadata(self):
        value = torch.ones(2, 3, 4, 4)
        record = {'batch': 0, 'input_rect': [160., 64., 192., 96.]}
        result = scatter_fixed_grid(value, [record, record], 1, (20, 40), (320, 640), 'mean')
        self.assertEqual(float(result.sum()), 48)
        for invalid in ({'batch': -1, 'input_rect': record['input_rect']},
                        {'batch': 1, 'input_rect': record['input_rect']},
                        {'batch': 0, 'input_rect': [0, 0, float('nan'), 32]},
                        {'batch': 0, 'input_rect': [32, 0, 16, 32]}):
            with self.assertRaises(ValueError):
                scatter_fixed_grid(value[:1], [invalid], 1, (20, 40), (320, 640))
        with self.assertRaises(ValueError):
            scatter_fixed_grid(value, [record], 1, (20, 40), (320, 640))
        with self.assertRaises(ValueError):
            scatter_fixed_grid(value, [record, record], 1, (40, 80), (320, 640))
        with self.assertRaises(ValueError):
            scatter_fixed_grid(value[..., :2, :2], [record, record], 1, (20, 40), (320, 640))

    def test_native_pixels_flip_and_actual_projected_border_rect(self):
        image = np.arange(100 * 150 * 3, dtype=np.uint8).reshape(100, 150, 3)
        sources = initial_sources('synthetic.png', (100, 150), (50, 75))
        sample = {KEY: sources}
        update_sources(sample, np.array([[-1, 0, 75], [0, 1, 0], [0, 0, 1]]), [0, 0, 75, 50])
        with patch('cv2.imread', return_value=image):
            crops, records, _, _ = build_crops(torch.zeros(1, 50, 75),
                                              [torch.tensor([[0., 45., 5., 50.]])], [sample[KEY]])
        np.testing.assert_equal(records[0]['source_rect'], [86, 36, 150, 100])
        np.testing.assert_equal(records[0]['input_rect'], [0, 18, 32, 50])
        expected = np.fliplr(image[36:100, 86:150])
        rgb = np.ascontiguousarray(expected[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
        std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        np.testing.assert_array_equal(crops[0].numpy(), (rgb - mean) / std)

    def test_input64_support_size_xy_values_and_gradient_all_heads(self):
        record = {'batch': 0, 'input_rect': [160., 64., 224., 128.]}
        for stride, size in ((8, 8), (16, 4), (32, 2)):
            value = torch.arange(48.).reshape(1, 3, 4, 4).requires_grad_()
            result = scatter_input_support(value, [record], 2,
                (320 // stride, 640 // stride), (320, 640))
            x0, y0 = 192 // stride - size // 2, 96 // stride - size // 2
            mode = 'area' if size < 4 else 'bilinear'
            options = {} if mode == 'area' else {'align_corners': False}
            expected = torch.nn.functional.interpolate(value, size=(size, size), mode=mode, **options)[0]
            torch.testing.assert_close(result[0, :, y0:y0+size, x0:x0+size], expected)
            self.assertEqual(float(result[1].sum()), 0)
            self.assertEqual(int(torch.count_nonzero(result)), int(torch.count_nonzero(expected)))
            result.sum().backward()
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue(torch.all(value.grad > 0))

    def test_input64_p5_origin_uses_half_two_not_half_four(self):
        record = {'batch': 0, 'input_rect': [160., 64., 224., 128.]}
        result = scatter_input_support(torch.ones(1, 1, 4, 4), [record], 1, (10, 20), (320, 640))
        expected = torch.zeros_like(result)
        expected[0, 0, 2:4, 5:7] = 1
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_input64_gain_independence_and_border_clipping(self):
        for stride in (8, 16, 32):
            value = torch.arange(48.).reshape(1, 3, 4, 4)
            record = {'batch': 0, 'input_rect': [160., 64., 224., 128.]}
            smaller = {'batch': 0, 'input_rect': [176., 80., 208., 112.]}
            arguments = (1, (320 // stride, 640 // stride), (320, 640))
            torch.testing.assert_close(scatter_input_support(value, [record], *arguments),
                                       scatter_input_support(value, [smaller], *arguments), rtol=0, atol=0)
            edge = {'batch': 0, 'input_rect': [-32., -32., 32., 32.]}
            result = scatter_input_support(value, [edge], *arguments)
            size = 64 // stride
            patch = resize_detail(value[0], (size, size))
            torch.testing.assert_close(result[0, :, :size//2, :size//2], patch[:, size//2:, size//2:])
            self.assertEqual(int(torch.count_nonzero(result)), int(torch.count_nonzero(patch[:, size//2:, size//2:])))

    def test_cam_zero_counters_distinguish_uncomputed_and_no_candidate_samples(self):
        model = SimpleNamespace(bgd_318_alpha_config={'track_cam_values': True}, training=True)
        boxes = [torch.ones(1, 4), torch.empty(0, 4)]
        cam = torch.zeros(2, 48, 112)
        cam[1, 20, 80] = 1
        update_cam_counts(model, boxes, cam)
        update_cam_counts(model, [torch.empty(0, 4)] * 2)
        values = model.direct_cam_counts['train']
        self.assertEqual(values, {'images': 4, 'candidate_images': 1, 'cam_computed_images': 2,
                                 'zero_cam_candidate_images': 1, 'no_candidate_images': 3})

    def test_cam_monitor_logs_rates_once_and_resets_distinct_ema_counters(self):
        from experiments.direct_geometry_monitor import reset_cam_epoch, log_cam_epoch
        with tempfile.TemporaryDirectory(prefix='direct_cam_monitor_') as folder:
            trainer = SimpleNamespace(model=SimpleNamespace(), ema=SimpleNamespace(ema=SimpleNamespace()),
                                      epoch=0, save_dir=Path(folder))
            reset_cam_epoch(trainer)
            self.assertIsNot(trainer.model.direct_cam_counts, trainer.ema.ema.direct_cam_counts)
            trainer.model.direct_cam_counts['train'] = {'candidate_images': 32, 'zero_cam_candidate_images': 4}
            trainer.ema.ema.direct_cam_counts['val'] = {'candidate_images': 32, 'zero_cam_candidate_images': 23}
            log_cam_epoch(trainer)
            log_cam_epoch(trainer)
            import json
            rows = (Path(folder) / 'cam_values.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])['val']['zero_cam_fraction_of_candidate_images'], 23 / 32)

    def test_original_grad_cam_can_be_zero_despite_nonzero_correct_gradient(self):
        activation = (-torch.ones(1, 3, 4, 7)).requires_grad_()
        classifier = torch.nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            classifier.weight.fill_(1.)
            classifier.bias.fill_(-2.)
        scores = classifier(activation).sigmoid()
        gradient, = torch.autograd.grad(scores.sum(), activation)
        expected = classifier.weight.reshape(1, 3, 1, 1) * scores * (1 - scores)
        torch.testing.assert_close(gradient, expected)
        self.assertGreater(float(gradient.abs().sum()), 0)
        cam = normalized_cam(activation, gradient, (64, 112))
        self.assertEqual(float(cam.sum()), 0)

    def test_no_crop_sample_is_batch_independent_and_regression_unchanged(self):
        raw = torch.randn(2, 65, 3, 7, requires_grad=True)
        logits = torch.randn(2, 1, 3, 7, requires_grad=True)
        output = replace_logits(raw, logits, [{'batch': 1}], 64)
        torch.testing.assert_close(output[0], raw[0], rtol=0, atol=0)
        torch.testing.assert_close(output[:, :64], raw[:, :64], rtol=0, atol=0)
        torch.testing.assert_close(output[1, 64:], logits[1], rtol=0, atol=0)
        output.sum().backward()
        self.assertEqual(float(logits.grad[0].abs().sum()), 0)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0)

    def test_deterministic_resize_matches_pytorch_values_and_gradients(self):
        for size in ((1, 1), (2, 3), (3, 2), (3, 3), (4, 4), (5, 7), (8, 8), (2, 7)):
            value = torch.randn(3, 4, 4, requires_grad=True)
            target = value.detach().clone().requires_grad_()
            actual = resize_detail(value, size)
            mode = 'area' if max(size) <= 4 else 'bilinear'
            options = {} if mode == 'area' else {'align_corners': False}
            expected = torch.nn.functional.interpolate(target[None], size=size, mode=mode, **options)[0]
            torch.testing.assert_close(actual, expected)
            weight = torch.randn_like(actual)
            (actual * weight).sum().backward()
            (expected * weight).sum().backward()
            torch.testing.assert_close(value.grad, target.grad)

    def test_metadata_batch_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            build_crops(torch.zeros(2, 64, 64), [torch.empty(0, 4)] * 2, [[]])

    def test_saved_detail_auxiliary_is_ema_not_raw_training_weights(self):
        from ultralytics.yolo.engine.trainer import BaseTrainer
        model = torch.nn.Module()
        model.detail_model = torch.nn.Conv2d(3, 3, 1)
        model.bgd_318_alpha_config = {'geometry_version': 5}
        ema = deepcopy(model).half()
        with torch.no_grad():
            model.detail_model.weight.add_(1.)
        with tempfile.TemporaryDirectory(prefix='direct_geometry_save_test_') as folder:
            trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema, updates=1),
                epoch=0, best_fitness=1., fitness=1., optimizer=torch.optim.AdamW(model.parameters()),
                args=SimpleNamespace(epochs=50), last=Path(folder) / 'last.pt',
                best=Path(folder) / 'best.pt', save_period=-1, wdir=Path(folder))
            BaseTrainer.save_model(trainer)
            checkpoint = torch.load(trainer.best, map_location='cpu', weights_only=False)
            for name, value in checkpoint['ema'].detail_model.state_dict().items():
                torch.testing.assert_close(value, checkpoint['detail_model'][name], rtol=0, atol=0)
            torch.testing.assert_close(checkpoint['model']['detail_model.weight'],
                                       model.detail_model.weight, rtol=0, atol=0)
            self.assertFalse(torch.equal(checkpoint['detail_model']['weight'].float(), model.detail_model.weight))


if __name__ == '__main__':
    unittest.main(verbosity=2)
