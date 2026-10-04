"""Opt-in, source-to-input geometry for native-resolution Direct crops.

Coordinates are continuous pixel *edges*, ordered xyxy. Matrices map source
edges to the actual augmented YOLO canvas. No image arrays enter metadata.
Only axis-aligned transforms are supported; unsupported transforms fail closed.
"""

from copy import deepcopy

import cv2
import numpy as np


KEY = 'direct_sources'


def transform_rect(rect, matrix):
    x0, y0, x1, y1 = rect
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=np.float64)
    projected = corners @ np.asarray(matrix, dtype=np.float64).T
    projected = projected[:, :2] / projected[:, 2:]
    return np.r_[projected.min(axis=0), projected.max(axis=0)]


def intersection(first, second):
    return np.array([max(first[0], second[0]), max(first[1], second[1]),
                     min(first[2], second[2]), min(first[3], second[3])], dtype=np.float64)


def area(rect):
    return max(0., rect[2] - rect[0]) * max(0., rect[3] - rect[1])


def assert_axis_aligned(matrix):
    matrix = np.asarray(matrix)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]) or \
            abs(matrix[0, 1]) > 1e-7 or abs(matrix[1, 0]) > 1e-7 or \
            abs(matrix[0, 0]) < 1e-8 or abs(matrix[1, 1]) < 1e-8:
        raise ValueError('Direct native crops require axis-aligned geometry; rotation/shear/perspective unsupported')


def update_sources(labels, matrix, clip):
    """Compose the *actual* image transform, preserving tile visibility."""
    if KEY not in labels:
        return
    assert_axis_aligned(matrix)
    output = []
    for source in labels[KEY]:
        source = deepcopy(source)
        source['matrix'] = np.asarray(matrix, dtype=np.float64) @ source['matrix']
        assert_axis_aligned(source['matrix'])
        source['visible'] = intersection(transform_rect(source['visible'], matrix), clip)
        if area(source['visible']) > 0:
            output.append(source)
    labels[KEY] = output


def initial_sources(path, original_hw, resized_hw):
    oh, ow = original_hw
    rh, rw = resized_hw
    return [{'path': str(path), 'shape': (oh, ow),
             'matrix': np.diag([rw / ow, rh / oh, 1.]).astype(np.float64),
             'visible': np.array([0, 0, rw, rh], dtype=np.float64), 'hsv': None}]


def legacy_letterbox_sources(batch, index):
    """Evaluation fallback for callers constructed before metadata was added.

    Training must never use this fallback: ratio_pad cannot describe Mosaic.
    """
    oh, ow = batch['ori_shape'][index]
    (gy, gx), (px, py) = batch['ratio_pad'][index]
    px, py = int(round(px - .1)), int(round(py - .1))
    matrix = np.array([[gx, 0, px], [0, gy, py], [0, 0, 1]], dtype=np.float64)
    return [{'path': batch['im_file'][index], 'shape': (oh, ow), 'matrix': matrix,
             'visible': transform_rect([0, 0, ow, oh], matrix), 'hsv': None}]


def native_crop(image, center, size=64):
    """Match the historical clamped fixed crop, returning its TRUE source rect."""
    height, width = image.shape[:2]
    pad_x, pad_y = max(0, size - width), max(0, size - height)
    left, top = pad_x // 2, pad_y // 2
    if pad_x or pad_y:
        image = cv2.copyMakeBorder(image, top, pad_y - top, left, pad_x - left,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    x0 = max(0, min(image.shape[1] - size, int(round(float(center[0]) + left - size / 2))))
    y0 = max(0, min(image.shape[0] - size, int(round(float(center[1]) + top - size / 2))))
    patch = image[y0:y0 + size, x0:x0 + size].copy()
    return patch, np.array([x0 - left, y0 - top, x0 - left + size, y0 - top + size], dtype=np.float64)


def apply_hsv(image, gains):
    if gains is None:
        return image
    x = np.arange(256, dtype=np.float64)
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    result = cv2.merge((cv2.LUT(hue, ((x * gains[0]) % 180).astype(np.uint8)),
                        cv2.LUT(sat, np.clip(x * gains[1], 0, 255).astype(np.uint8)),
                        cv2.LUT(val, np.clip(x * gains[2], 0, 255).astype(np.uint8))))
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
