"""Geometry-corrected single-head Direct, without alpha or confidence weights.

Keep legacy checkpoints on their old runtime. This v5 path fixes CAM H/W,
augmentation-aware native crops, true crop footprint scatter and CAM gradient
isolation. YOLO/Detail/fusion layer shapes are unchanged.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.yolo.data.direct_geometry import (
    KEY, area, assert_axis_aligned, intersection, legacy_letterbox_sources,
    native_crop, transform_rect, apply_hsv,
)


METHOD = 'best318_framework_corrected_single_head_direct_geometry_v5'
METHOD_FIXED_GRID = METHOD + '_fixed4'
METHOD_INPUT_SUPPORT = METHOD + '_input64'
METHOD_MULTI_INPUT_SUPPORT = 'best318_framework_multi_head_direct_geometry_v5_input64'


def aggregate_head_cams(activations, gradients, input_hw, cam_mode='Grad'):
    """Preserve legacy multi-head CAM: grid-normalize, resize, mean, normalize.

    Do not normalize each resized head a second time before the mean: this
    would change the relative contribution of heads with different grids.
    This aggregation is unrelated to the sum reduction of overlapping crops.
    """
    if len(activations) != len(gradients) or not activations:
        raise ValueError('Multi-head CAM needs matching nonempty activation/gradient lists')
    head_cams = []
    for activation, gradient in zip(activations, gradients):
        act, grad = activation.detach().float(), gradient.detach().float()
        if not torch.isfinite(act).all() or not torch.isfinite(grad).all():
            raise FloatingPointError('Non-finite multi-head CAM activation/gradient')
        if cam_mode == 'Grad':
            weights = grad.mean((-2, -1), keepdim=True)
        elif cam_mode == 'GradCAM':
            weights = (grad * act / (act.sum((-2, -1), keepdim=True) + 1e-7)).sum((-2, -1), keepdim=True)
        else:
            raise ValueError(f'Unsupported historical CAM mode: {cam_mode}')
        cam = (weights * act).sum(1, keepdim=True).clamp_min(0)
        cam = cam - cam.amin((-2, -1), keepdim=True)
        cam = cam / (cam.amax((-2, -1), keepdim=True) + 1e-7)
        head_cams.append(F.interpolate(cam, size=tuple(input_hw), mode='bilinear', align_corners=False)[:, 0])
    aggregate = torch.stack(head_cams).mean(0)
    aggregate = aggregate - aggregate.amin((-2, -1), keepdim=True)
    aggregate = aggregate / (aggregate.amax((-2, -1), keepdim=True) + 1e-7)
    return aggregate, head_cams


def normalized_cam(activation, gradient, input_hw, cam_mode='Grad'):
    """Resize using explicit H/W; avoid OpenCV's reversed size convention."""
    activation, gradient = activation.detach().float(), gradient.detach().float()
    if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError('Non-finite CAM activation/gradient')
    if cam_mode == 'Grad':
        weights = gradient.mean(dim=(-2, -1), keepdim=True)
    elif cam_mode == 'GradCAM':  # historical alternative, not used in these runs
        weights = (gradient * activation / (activation.sum((-2, -1), keepdim=True) + 1e-7)).sum(
            (-2, -1), keepdim=True)
    else:
        raise ValueError(f'Unsupported historical CAM mode: {cam_mode}')
    cam = (weights * activation).sum(dim=1, keepdim=True).clamp_min(0)
    cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
    cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-7)
    cam = F.interpolate(cam, size=tuple(input_hw), mode='bilinear', align_corners=False)
    # Historical aggregation normalizes once more after resizing.
    cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
    cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-7)
    return cam[:, 0]


def centers_for_sources(cam, boxes, sources, crop_size=64):
    """All-ones window fully inside both candidate and visible source tile.

    Pick the source containing the box center, or the largest visible overlap.
    Tiny boxes / zero CAM fall back to the clipped box center, never a fabricated
    hotspot. None marks candidates wholly in padding (no native image there).
    """
    height, width = cam.shape
    output, owners, cache, windows = [], [], {}, []
    for box in boxes.detach().float().cpu().numpy():
        box = intersection(box, [0, 0, width, height])
        bx, by = (box[:2] + box[2:]) / 2
        owner = next((source for source in sources
                      if source['visible'][0] <= bx < source['visible'][2]
                      and source['visible'][1] <= by < source['visible'][3]), None)
        if owner is None:
            owner = max(sources, key=lambda item: area(intersection(box, item['visible'])), default=None)
        if owner is None or area(intersection(box, owner['visible'])) <= 0:
            output.append(None)
            owners.append(None)
            windows.append(None)
            continue
        matrix = owner['matrix']
        assert_axis_aligned(matrix)
        kh = max(1, int(round(crop_size * abs(matrix[1, 1]))))
        kw = max(1, int(round(crop_size * abs(matrix[0, 0]))))
        rect = intersection(box, owner['visible'])
        center = (rect[:2] + rect[2:]) / 2
        x0, y0 = int(math.ceil(rect[0])), int(math.ceil(rect[1]))
        x1, y1 = int(math.floor(rect[2])), int(math.floor(rect[3]))
        if kw <= width and kh <= height and x1 - x0 >= kw and y1 - y0 >= kh:
            if (kh, kw) not in cache:
                kernel = torch.ones((1, 1, kh, kw), device=cam.device, dtype=torch.float32)
                # Outer training AMP must not round CAM window sums to FP16
                # while validation searches them in FP32.
                with torch.autocast(device_type=cam.device.type, enabled=False):
                    cache[kh, kw] = F.conv2d(cam[None, None].float(), kernel)[0, 0]
            response = cache[kh, kw][y0:y1 - kh + 1, x0:x1 - kw + 1]
            if response.numel() and response.max().item() > 0:
                offset = int(response.argmax().item())
                dy, dx = divmod(offset, response.shape[1])
                center = np.array([x0 + dx + kw / 2, y0 + dy + kh / 2])
        output.append(np.asarray(center, dtype=np.float64))
        owners.append(owner)
        windows.append((kh, kw))
    return output, owners, windows


def build_crops(cam, boxes_by_image, sources_by_image):
    """Return normalized RGB 64² inputs and one exact geometry record per crop."""
    if cam.ndim != 3 or cam.shape[0] != len(boxes_by_image) or cam.shape[0] != len(sources_by_image):
        raise ValueError('CAM / candidate / source metadata batch lengths differ')
    tensors, records, centers_by_image, windows_by_image, image_cache = [], [], [], [], {}
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    for index, (boxes, sources) in enumerate(zip(boxes_by_image, sources_by_image)):
        centers, owners, windows = centers_for_sources(cam[index], boxes, sources)
        centers_by_image.append(centers)
        windows_by_image.append(windows)
        for candidate, (center, source) in enumerate(zip(centers, owners)):
            if source is None:
                continue
            path = source['path']
            if path not in image_cache:
                image_cache[path] = cv2.imread(path)
                if image_cache[path] is None:
                    raise FileNotFoundError(f'Cannot read Direct crop source: {path}')
            image = image_cache[path]
            if image.shape[:2] != tuple(source['shape']):
                raise ValueError(f'Crop source size disagrees with metadata: {path}')
            matrix = source['matrix']
            inverse = np.linalg.inv(matrix)
            source_center = (inverse @ np.r_[center, 1])[:2]
            patch, source_rect = native_crop(image, source_center)
            input_rect = transform_rect(source_rect, matrix)
            visible = intersection(input_rect, source['visible'])
            visible = intersection(visible, [0, 0, cam.shape[-1], cam.shape[-2]])
            if area(visible) <= 0:
                continue
            # Mask source pixels removed by Mosaic / affine canvas clipping.
            # Padding outside the original source is already black.
            source_visible = transform_rect(visible, inverse)
            xs = source_rect[0] + np.arange(64) + .5
            ys = source_rect[1] + np.arange(64) + .5
            mask = ((ys[:, None] >= source_visible[1]) & (ys[:, None] < source_visible[3]) &
                    (xs[None, :] >= source_visible[0]) & (xs[None, :] < source_visible[2]))
            patch = apply_hsv(patch, source['hsv'])
            patch[~mask] = 0
            if matrix[0, 0] < 0:
                patch = np.fliplr(patch)
            if matrix[1, 1] < 0:
                patch = np.flipud(patch)
            rgb = np.ascontiguousarray(patch[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.
            tensors.append(torch.from_numpy((rgb - mean) / std))
            records.append({'batch': index, 'candidate': candidate, 'path': path,
                            'source_rect': source_rect.tolist(), 'input_rect': input_rect.tolist(),
                            'visible_rect': visible.tolist(), 'center_xy': center.tolist(),
                            'matrix': matrix.tolist()})
    if not tensors:
        return None, records, centers_by_image, windows_by_image
    return torch.stack(tensors), records, centers_by_image, windows_by_image


def resize_weights(input_size, output_size, mode, like):
    """Separable resize weights: deterministic GEMM backward, no atomic pool.

    Area matches PyTorch adaptive-average bins. Bilinear matches
    align_corners=False with border replication. Detail has only 4x4 cells.
    """
    weights = torch.zeros((output_size, input_size), dtype=torch.float32)
    for index in range(output_size):
        if mode == 'area':
            start = math.floor(index * input_size / output_size)
            stop = math.ceil((index + 1) * input_size / output_size)
            weights[index, start:stop] = 1. / (stop - start)
        elif mode == 'bilinear':
            position = max(0., (index + .5) * input_size / output_size - .5)
            first = min(input_size - 1, math.floor(position))
            second = min(input_size - 1, first + 1)
            fraction = position - first
            weights[index, first] += 1. - fraction
            weights[index, second] += fraction
        else:
            raise ValueError(f'Unsupported deterministic resize: {mode}')
    return weights.to(device=like.device, dtype=like.dtype)


def resize_detail(value, size, cache=None):
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    ph, pw = size
    ih, iw = value.shape[-2:]
    mode = 'area' if ph <= ih and pw <= iw else 'bilinear'
    key = (ih, iw, ph, pw, mode)
    if cache is None:
        cache = {}
    if key not in cache:
        cache[key] = (resize_weights(ih, ph, mode, value), resize_weights(iw, pw, mode, value))
    wy, wx = cache[key]
    return wy @ value @ wx.T


def scatter_crops(detail_outputs, records, batch_size, feature_hw, input_hw, reduce='sum'):
    """Rasterize the TRUE projected crop rectangle, BCHW[y,x], differentiably.

    Cell bounds are floor(start/stride), ceil(end/stride): <1-cell quantization.
    Resize the full crop first, then clip to visible cells; clipping must not
    stretch a partial crop across its entire original footprint.
    """
    if reduce not in {'sum', 'mean'} or len(detail_outputs) != len(records):
        raise ValueError('Invalid crop scatter reduction/count')
    fh, fw = feature_hw
    sy, sx = input_hw[0] / fh, input_hw[1] / fw
    feature = detail_outputs.new_zeros((batch_size, detail_outputs.shape[1], fh, fw))
    counts = detail_outputs.new_zeros((batch_size, 1, fh, fw)) if reduce == 'mean' else None
    resize_cache = {}
    for value, record in zip(detail_outputs, records):
        x0, y0, x1, y1 = record['input_rect']
        fx0, fy0 = math.floor(x0 / sx), math.floor(y0 / sy)
        fx1, fy1 = math.ceil(x1 / sx), math.ceil(y1 / sy)
        vx0, vy0, vx1, vy1 = record['visible_rect']
        tx0, ty0 = max(0, fx0, math.floor(vx0 / sx)), max(0, fy0, math.floor(vy0 / sy))
        tx1, ty1 = min(fw, fx1, math.ceil(vx1 / sx)), min(fh, fy1, math.ceil(vy1 / sy))
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        ph, pw = fy1 - fy0, fx1 - fx0
        if ph <= 0 or pw <= 0:
            raise ValueError('Non-positive projected crop footprint')
        patch = resize_detail(value, (ph, pw), resize_cache)
        batch = record['batch']
        feature[batch, :, ty0:ty1, tx0:tx1].add_(patch[:, ty0 - fy0:ty1 - fy0, tx0 - fx0:tx1 - fx0])
        if counts is not None:
            counts[batch, :, ty0:ty1, tx0:tx1].add_(1.)
    if counts is not None:
        feature = feature / counts.clamp_min(1)
    return feature


def scatter_fixed_grid(detail_outputs, records, batch_size, feature_hw, input_hw, reduce='sum',
                       return_records=False):
    """Inject the unchanged 4x4 Detail grid around its actual crop center.

    This is fixed-context feature injection, not crop-footprint rasterization.
    Original-image gain changes crop coordinates, but never this grid's size.
    Clip at the input canvas without shifting or stretching a partial patch.
    """
    if tuple(detail_outputs.shape[-2:]) != (4, 4):
        raise ValueError('Fixed-grid injection requires the original 4x4 Detail output')
    fh, fw = feature_hw
    sy, sx = input_hw[0] / fh, input_hw[1] / fw
    if not (math.isclose(sy, sx) and any(math.isclose(sx, stride) for stride in (16., 32.))):
        raise ValueError('Fixed-grid injection currently requires the P4/P5 feature grid')
    return _scatter_centered(detail_outputs, records, batch_size, feature_hw, input_hw,
                             (4, 4), reduce, return_records)


def scatter_input_support(detail_outputs, records, batch_size, feature_hw, input_hw, reduce='sum',
                          return_records=False):
    """Fixed 64-input-pixel context: P3=8x8, P4=4x4, P5=2x2.

    Original crop gain affects the actual center, not the injection footprint.
    Original Detail output remains 4x4; resizing happens only before scatter.
    """
    fh, fw = feature_hw
    sy, sx = input_hw[0] / fh, input_hw[1] / fw
    if not (math.isclose(sy, sx) and any(math.isclose(sx, stride) for stride in (8., 16., 32.))):
        raise ValueError('Input-64 injection requires the P3/P4/P5 feature grid')
    size = (max(1, int(round(64 / sy))), max(1, int(round(64 / sx))))
    return _scatter_centered(detail_outputs, records, batch_size, feature_hw, input_hw,
                             size, reduce, return_records)


def _scatter_centered(detail_outputs, records, batch_size, feature_hw, input_hw, patch_hw,
                      reduce, return_records):
    if reduce not in {'sum', 'mean'} or len(detail_outputs) != len(records):
        raise ValueError('Invalid centered scatter reduction/count')
    if detail_outputs.ndim != 4 or tuple(detail_outputs.shape[-2:]) != (4, 4):
        raise ValueError('Centered scatter requires the original NCHW 4x4 Detail output')
    fh, fw = feature_hw
    sy, sx = input_hw[0] / fh, input_hw[1] / fw
    ph, pw = patch_hw
    feature = detail_outputs.new_zeros((batch_size, detail_outputs.shape[1], fh, fw))
    counts = detail_outputs.new_zeros((batch_size, 1, fh, fw)) if reduce == 'mean' else None
    written_records = []
    resize_cache = {}
    for value, record in zip(detail_outputs, records):
        x0, y0, x1, y1 = record['input_rect']
        if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
            raise ValueError('Invalid actual crop rectangle in fixed-grid scatter')
        batch = record['batch']
        if not isinstance(batch, (int, np.integer)) or not 0 <= batch < batch_size:
            raise ValueError('Invalid crop batch index in fixed-grid scatter')
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        gx0, gy0 = int(round(center_x / sx)) - pw // 2, int(round(center_y / sy)) - ph // 2
        tx0, ty0 = max(0, gx0), max(0, gy0)
        tx1, ty1 = min(fw, gx0 + pw), min(fh, gy0 + ph)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        patch = resize_detail(value, (ph, pw), resize_cache)
        feature[batch, :, ty0:ty1, tx0:tx1].add_(
            patch[:, ty0 - gy0:ty1 - gy0, tx0 - gx0:tx1 - gx0])
        if counts is not None:
            counts[batch, :, ty0:ty1, tx0:tx1].add_(1.)
        written_records.append(record)
    if counts is not None:
        feature = feature / counts.clamp_min(1)
    return (feature, written_records) if return_records else feature


def update_cam_counts(model, boxes, cam=None):
    """Small integer diagnostics only; never replace CAM or touch gradients."""
    if not model.bgd_318_alpha_config.get('track_cam_values', False):
        return
    if not hasattr(model, 'direct_cam_counts'):
        model.direct_cam_counts = {}
    key = 'train' if model.training else 'val'
    counts = model.direct_cam_counts.setdefault(key, {'images': 0, 'candidate_images': 0,
        'cam_computed_images': 0, 'zero_cam_candidate_images': 0, 'no_candidate_images': 0})
    has_candidates = [bool(len(value)) for value in boxes]
    counts['images'] += len(boxes)
    counts['candidate_images'] += sum(has_candidates)
    counts['no_candidate_images'] += len(boxes) - sum(has_candidates)
    if cam is not None:
        zero = (cam.amax(dim=(-2, -1)) <= 0).tolist()
        counts['cam_computed_images'] += len(boxes)
        counts['zero_cam_candidate_images'] += sum(flag and empty for flag, empty in zip(has_candidates, zero))


def decode_raw(detect, raw):
    from ultralytics.yolo.utils.tal import make_anchors, dist2bbox
    shape = raw[0].shape
    detect.anchors, detect.strides = (x.transpose(0, 1) for x in make_anchors(raw, detect.stride, .5))
    detect.shape = shape
    joined = torch.cat([value.view(shape[0], detect.no, -1) for value in raw], dim=2)
    box, cls = joined.split((detect.reg_max * 4, detect.nc), dim=1)
    decoded_box = dist2bbox(detect.dfl(box), detect.anchors[None], xywh=True, dim=1) * detect.strides
    return torch.cat((decoded_box, cls.sigmoid()), dim=1)


def replace_logits(raw_level, fused_logits, records, regression_channels):
    """A sample without native crops stays YOLO-only, regardless of its peers."""
    active = torch.zeros(raw_level.shape[0], dtype=torch.bool, device=raw_level.device)
    for record in records:
        active[record['batch']] = True
    logits = torch.where(active[:, None, None, None], fused_logits, raw_level[:, regression_channels:])
    return torch.cat((raw_level[:, :regression_channels], logits), dim=1)


def detached_prediction(prediction):
    """Metric buffers must never retain the internally enabled CAM graph."""
    decoded, raw = prediction
    return decoded.detach(), [value.detach() for value in raw]


def geometry_predict_once(model, images, profile=False, visualize=False):
    """No parameter gradient clearing, no model deepcopy, no retained batches."""
    if model.bgd_318_alpha_config.get('geometry_multi_head', False):
        return geometry_multi_predict_once(model, images, profile, visualize)
    from ultralytics.nn.tasks import BaseModel
    config = model.bgd_318_alpha_config
    wrapper = model.bgd_318_original_predict_once.__self__
    # EMA/device moves may deserialize or copy the wrapper: always rebind.
    wrapper.model, wrapper.detail_model = model, model.detail_model
    wrapper.batch = None
    detect, detail = model.model[-1], model.detail_model
    level = {'high': 0, 'mid': 1, 'low': 2}[config['head_select']]
    if detect.nc != 1 or detect.export:
        raise ValueError('Geometry v5 currently requires the single-class non-export Detect path')
    batch = getattr(model, 'batch', None)
    capture, handles = [], []
    diagnostic = bool(getattr(model, 'direct_geometry_audit', False))
    try:
        if batch is not None:
            handles.append(detect.cv3[level][1].register_forward_hook(lambda _m, _i, out: capture.append(out)))
        base = BaseModel._predict_once(model, images, profile, visualize)
        if batch is None:
            return base if model.training else detached_prediction(base)
        raw = base if model.training else base[1]
        decoded = decode_raw(detect, raw) if model.training else base[0]
        if len(capture) != 1:
            raise RuntimeError('Selected Detect hook did not capture exactly one activation')
        activation = capture[0]
        boxes_by_image = []
        from ultralytics.yolo.utils.ops import xywh2xyxy
        for value in decoded.detach():
            selected = value[4] > float(config['candidate_conf'])
            boxes_by_image.append(xywh2xyxy(value[:4, selected].T))
        if not any(len(boxes) for boxes in boxes_by_image):
            update_cam_counts(model, boxes_by_image)
            if diagnostic:
                model.direct_geometry_audit_result = {'no_candidates': True, 'input_hw': list(images.shape[-2:])}
            return base if model.training else detached_prediction(base)
        if KEY in batch:
            sources = batch[KEY]
        elif model.training:
            raise RuntimeError('Training Direct crops lack augmentation-aware source metadata; refusing unsafe fallback')
        else:
            sources = [legacy_letterbox_sources(batch, index) for index in range(len(images))]
        # Equivalent historical target: sum of ALL original sigmoid cls scores.
        # autograd.grad touches only the requested activation, never .grad on
        # parameters, so optimizer gradient accumulation remains intact.
        gradient, = torch.autograd.grad(decoded[:, 4:].sum(), activation, retain_graph=model.training)
        cam = normalized_cam(activation, gradient, images.shape[-2:], wrapper.cam_mode)
        update_cam_counts(model, boxes_by_image, cam)
        if not model.training:
            activation = activation.detach()
            raw = [value.detach() for value in raw]
            decoded = decoded.detach()
            base = None
            capture.clear()
        if tuple(cam.shape) != (len(images), *images.shape[-2:]):
            raise RuntimeError('CAM shape does not match the actual YOLO input canvas')
        with torch.no_grad():
            crops, records, centers, windows = build_crops(cam, boxes_by_image, sources)
        if crops is None:
            if diagnostic:
                model.direct_geometry_audit_result = {'no_crops': True, 'input_hw': list(images.shape[-2:]),
                                                     'cam_hw': list(cam.shape[-2:]), 'records': []}
            return raw if model.training else (decoded, raw)
        detail.train(model.training)
        crops = crops.to(device=images.device, dtype=next(detail.parameters()).dtype, non_blocking=True)
        outputs = detail(crops)
        scatter_mode = config.get('geometry_scatter_mode', 'projected')
        if scatter_mode == 'fixed_grid':
            if level not in (1, 2):
                raise ValueError('Fixed 4x4 injection is enabled only for the P4/P5 heads')
            feature, fusion_records = scatter_fixed_grid(
                outputs, records, len(images), activation.shape[-2:], images.shape[-2:],
                config.get('scatter_reduce', 'sum'), return_records=True)
        elif scatter_mode == 'input64':
            feature, fusion_records = scatter_input_support(
                outputs, records, len(images), activation.shape[-2:], images.shape[-2:],
                config.get('scatter_reduce', 'sum'), return_records=True)
        elif scatter_mode == 'projected':
            feature = scatter_crops(outputs, records, len(images), activation.shape[-2:], images.shape[-2:],
                                    config.get('scatter_reduce', 'sum'))
            fusion_records = records
        else:
            raise ValueError(f'Unsupported geometry scatter mode: {scatter_mode}')
        joined = torch.cat((feature.to(activation.dtype), activation), dim=1)
        joined = detail.relu_for_yolo(detail.bn_for_yolo(detail.conv_for_yolo(
            joined.to(detail.conv_for_yolo.weight.dtype))))
        logits = detect.cv3[level][2](joined.to(detect.cv3[level][2].weight.dtype))
        # Fresh tensors avoid mutating base values still needed for backprop.
        fused_raw = list(raw)
        fused_raw[level] = replace_logits(raw[level], logits, fusion_records, detect.reg_max * 4)
        if diagnostic:
            model.direct_geometry_audit_result = {
                'input_hw': list(images.shape[-2:]), 'cam_hw': list(cam.shape[-2:]),
                'activation_hw': list(activation.shape[-2:]), 'detail_hw': list(outputs.shape[-2:]),
                'scatter_mode': scatter_mode,
                'records': records, 'fusion_records': fusion_records,
                'windows': windows, 'cam': cam.detach().cpu(),
                'crops': crops.detach().float().cpu(), 'base_decoded': decoded.detach().cpu(),
                'fused_decoded': decode_raw(detect, fused_raw).detach().cpu(),
                'scatter': feature.detach().float().cpu(),
            }
        return fused_raw if model.training else detached_prediction((decode_raw(detect, fused_raw), fused_raw))
    finally:
        for handle in handles:
            handle.remove()
        capture.clear()
        wrapper.batch = None
        wrapper.activations.clear()
        wrapper.gradients.clear()
        wrapper.remove_hook()


def geometry_multi_predict_once(model, images, profile=False, visualize=False):
    """One YOLO/CAM/crop/Detail pass, independent classification fusion per head.

    Uses exactly the single-head v5 source/crop/center/scatter helpers. Regression
    stays unchanged at every level. All temporary hooks are removed in finally.
    """
    from ultralytics.nn.tasks import BaseModel
    from ultralytics.yolo.utils.ops import xywh2xyxy
    config = model.bgd_318_alpha_config
    if config.get('geometry_scatter_mode') != 'input64' or config.get('scatter_reduce') != 'sum':
        raise ValueError('Corrected multi-head v5 requires input64 support and sum')
    names = list(config['head_selects'])
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Corrected multi-head v5 needs distinct heads')
    levels = [{'high': 0, 'mid': 1, 'low': 2}[name] for name in names]
    wrapper = model.bgd_318_original_predict_once.__self__
    wrapper.model, wrapper.detail_model, wrapper.batch = model, model.detail_model, None
    detect, detail = model.model[-1], model.detail_model
    if detect.nc != 1 or detect.export:
        raise ValueError('Geometry v5 requires the single-class non-export Detect path')
    batch = getattr(model, 'batch', None)
    captures, handles = {level: [] for level in levels}, []
    diagnostic = bool(getattr(model, 'direct_geometry_audit', False))
    try:
        if batch is not None:
            for level in levels:
                handles.append(detect.cv3[level][1].register_forward_hook(
                    lambda _m, _i, out, selected=level: captures[selected].append(out)))
        base = BaseModel._predict_once(model, images, profile, visualize)
        if batch is None:
            return base if model.training else detached_prediction(base)
        raw = base if model.training else base[1]
        decoded = decode_raw(detect, raw) if model.training else base[0]
        if any(len(captures[level]) != 1 for level in levels):
            raise RuntimeError('Multi-head hooks did not capture exactly one activation per head')
        activations = [captures[level][0] for level in levels]
        boxes = [xywh2xyxy(value[:4, value[4] > float(config['candidate_conf'])].T)
                 for value in decoded.detach()]
        if not any(len(value) for value in boxes):
            update_cam_counts(model, boxes)
            if diagnostic:
                model.direct_geometry_audit_result = {'no_candidates': True, 'input_hw': list(images.shape[-2:])}
            return base if model.training else detached_prediction(base)
        if KEY in batch:
            sources = batch[KEY]
        elif model.training:
            raise RuntimeError('Training multi-head Direct lacks augmentation-aware source metadata')
        else:
            sources = [legacy_letterbox_sources(batch, i) for i in range(len(images))]
        gradients = torch.autograd.grad(decoded[:, 4:].sum(), tuple(activations), retain_graph=model.training)
        cam, head_cams = aggregate_head_cams(activations, gradients, images.shape[-2:], wrapper.cam_mode)
        update_cam_counts(model, boxes, cam)
        if not model.training:
            activations = [value.detach() for value in activations]
            raw, decoded, base = [value.detach() for value in raw], decoded.detach(), None
            for values in captures.values():
                values.clear()
        if tuple(cam.shape) != (len(images), *images.shape[-2:]):
            raise RuntimeError('Aggregated CAM does not match the actual input canvas')
        with torch.no_grad():
            crops, records, centers, windows = build_crops(cam, boxes, sources)
        if crops is None:
            if diagnostic:
                model.direct_geometry_audit_result = {'no_crops': True, 'input_hw': list(images.shape[-2:]),
                    'cam_hw': list(cam.shape[-2:]), 'cam': cam.detach().cpu(), 'records': []}
            return raw if model.training else detached_prediction((decoded, raw))
        detail.train(model.training)
        crops = crops.to(device=images.device, dtype=next(detail.parameters()).dtype, non_blocking=True)
        outputs = detail(crops)  # shared Detail encoder called exactly once
        fused_raw, head_results = list(raw), {}
        for slot, (name, level, activation) in enumerate(zip(names, levels, activations)):
            feature, fusion_records = scatter_input_support(outputs, records, len(images),
                activation.shape[-2:], images.shape[-2:], 'sum', return_records=True)
            conv, bn, relu = detail.conv_for_yolo_multi[slot], detail.bn_for_yolo_multi[slot], detail.relu_for_yolo_multi[slot]
            joined = torch.cat((feature.to(activation.dtype), activation), dim=1)
            joined = relu(bn(conv(joined.to(conv.weight.dtype))))
            classifier = detect.cv3[level][2]
            logits = classifier(joined.to(classifier.weight.dtype))
            fused_raw[level] = replace_logits(raw[level], logits, fusion_records, detect.reg_max * 4)
            if diagnostic:
                head_results[name] = {'activation_hw': list(activation.shape[-2:]),
                                      'scatter': feature.detach().float().cpu(), 'fusion_records': fusion_records}
        if diagnostic:
            model.direct_geometry_audit_result = {'input_hw': list(images.shape[-2:]),
                'cam_hw': list(cam.shape[-2:]), 'detail_hw': list(outputs.shape[-2:]), 'scatter_mode': 'input64',
                'records': records, 'windows': windows, 'cam': cam.detach().cpu(),
                'cam_per_head_before_aggregate': [value.detach().cpu() for value in head_cams],
                'crops': crops.detach().float().cpu(), 'heads': head_results,
                'base_decoded': decoded.detach().cpu(), 'fused_decoded': decode_raw(detect, fused_raw).detach().cpu()}
        return fused_raw if model.training else detached_prediction((decode_raw(detect, fused_raw), fused_raw))
    finally:
        for handle in handles:
            handle.remove()
        for values in captures.values():
            values.clear()
        wrapper.batch = None
        wrapper.activations.clear()
        wrapper.gradients.clear()
        wrapper.remove_hook()
