"""Export real P4 Grad-CAM/Detail/fusion assets for a paper architecture figure."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from run_318_fusion_alpha import install_legacy_modules, load_joint_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = (
    ROOT
    / "experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/best.pt"
)
DEFAULT_IMAGE = Path("/home/user/projects/czy/mydata/images/test/01979.jpg")
DEFAULT_LABEL = Path("/home/user/projects/czy/mydata/labels/test/01979.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--archive", type=Path, default=ROOT / "what is code.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def letterbox(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    resized_width = int(round(width * gain))
    resized_height = int(round(height * gain))
    pad_width = (size - resized_width) / 2
    pad_height = (size - resized_height) / 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left, right = int(round(pad_width - 0.1)), int(round(pad_width + 0.1))
    top, bottom = int(round(pad_height - 0.1)), int(round(pad_height + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, gain, pad_width, pad_height, (resized_width, resized_height)


def read_ground_truth(label_path: Path, width: int, height: int) -> np.ndarray:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _class_id, cx, cy, box_width, box_height = map(float, line.split())
        rows.append(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
        )
    if not rows:
        raise RuntimeError(f"No ground-truth boxes in {label_path}")
    return np.asarray(rows, dtype=np.float32)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    intersection_min = np.maximum(box[:2], boxes[:, :2])
    intersection_max = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.maximum(intersection_max - intersection_min, 0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def to_original_box(box: np.ndarray, gain: float, pad_width: float, pad_height: float):
    result = box.astype(np.float32).copy()
    result[(0, 2),] = (result[(0, 2),] - pad_width) / gain
    result[(1, 3),] = (result[(1, 3),] - pad_height) / gain
    return result


def to_original_point(point: np.ndarray, gain: float, pad_width: float, pad_height: float):
    return np.asarray(
        [(point[0] - pad_width) / gain, (point[1] - pad_height) / gain],
        dtype=np.float32,
    )


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum, maximum = float(array.min()), float(array.max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - minimum) / (maximum - minimum) * 255, 0, 255).astype(np.uint8)


def heatmap(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    normalized = normalize_uint8(array)
    resized = cv2.resize(normalized, size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(resized, cv2.COLORMAP_VIRIDIS)


def add_title(image: np.ndarray, title: str, width: int = 420, height: int = 300):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    available_height = height - 42
    scale = min(width / image.shape[1], available_height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
    )
    x = (width - resized.shape[1]) // 2
    y = 40 + (available_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
    return canvas


def save_visuals(
    output_dir: Path,
    original_bgr: np.ndarray,
    ground_truth: np.ndarray,
    candidate_box: np.ndarray,
    centre: np.ndarray,
    crop_rgb: np.ndarray,
    cam_letterbox: np.ndarray,
    detail_feature: np.ndarray,
    p4_activation: np.ndarray,
    fused_activation: np.ndarray,
    final_boxes: np.ndarray,
    gain: float,
    pad_width: float,
    pad_height: float,
    resized_size: tuple[int, int],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = original_bgr.shape[:2]
    cv2.imwrite(str(output_dir / "01_input_image.jpg"), original_bgr)

    annotated = original_bgr.copy()
    gt = tuple(int(round(value)) for value in ground_truth)
    pred = tuple(int(round(value)) for value in candidate_box)
    cv2.rectangle(annotated, gt[:2], gt[2:], (46, 180, 46), 5)
    cv2.rectangle(annotated, pred[:2], pred[2:], (34, 34, 220), 5)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    crop_box = (
        max(0, cx - 32), max(0, cy - 32), min(width - 1, cx + 32), min(height - 1, cy + 32)
    )
    cv2.rectangle(annotated, crop_box[:2], crop_box[2:], (220, 210, 20), 5)
    cv2.circle(annotated, (cx, cy), 9, (0, 230, 255), -1)
    cv2.putText(annotated, "Ground truth", (gt[0], max(28, gt[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 180, 46), 2)
    cv2.putText(annotated, "Candidate", (pred[0], min(height - 12, pred[1] + 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 34, 220), 2)
    cv2.imwrite(str(output_dir / "02_candidate_and_detail_crop.jpg"), annotated)

    resized_width, resized_height = resized_size
    left, top = int(round(pad_width - 0.1)), int(round(pad_height - 0.1))
    cam_unpadded = cam_letterbox[top : top + resized_height, left : left + resized_width]
    cam_original = cv2.resize(cam_unpadded, (width, height), interpolation=cv2.INTER_LINEAR)
    colored_cam = cv2.applyColorMap(normalize_uint8(cam_original), cv2.COLORMAP_JET)
    cam_overlay = cv2.addWeighted(original_bgr, 0.52, colored_cam, 0.48, 0)
    cv2.rectangle(cam_overlay, pred[:2], pred[2:], (255, 255, 255), 4)
    cv2.circle(cam_overlay, (cx, cy), 9, (0, 255, 255), -1)
    cv2.imwrite(str(output_dir / "03_p4_gradcam_overlay.jpg"), cam_overlay)

    restricted_cam = np.zeros_like(cam_original)
    x1, y1, x2, y2 = pred
    restricted_cam[y1:y2, x1:x2] = cam_original[y1:y2, x1:x2]
    restricted_color = cv2.applyColorMap(normalize_uint8(restricted_cam), cv2.COLORMAP_JET)
    restricted_overlay = cv2.addWeighted(original_bgr, 0.52, restricted_color, 0.48, 0)
    cv2.rectangle(restricted_overlay, pred[:2], pred[2:], (255, 255, 255), 4)
    cv2.circle(restricted_overlay, (cx, cy), 9, (0, 255, 255), -1)
    cv2.imwrite(
        str(output_dir / "03b_candidate_restricted_gradcam.jpg"), restricted_overlay
    )

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "04_detail_crop_64.png"), crop_bgr)
    crop_preview = cv2.resize(crop_bgr, (384, 384), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "05_detail_crop_preview.png"), crop_preview)

    channel_panels = []
    for index, channel in enumerate(detail_feature):
        panel = heatmap(channel, (240, 240))
        cv2.putText(panel, f"Route channel {index + 1}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        channel_panels.append(panel)
    detail_channels = np.concatenate(channel_panels, axis=1)
    cv2.imwrite(str(output_dir / "06_detail_feature_channels.png"), detail_channels)

    detail_rgb = np.transpose(detail_feature, (1, 2, 0))
    detail_rgb = normalize_uint8(detail_rgb)
    detail_rgb = cv2.resize(detail_rgb, (384, 384), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "07_detail_feature_rgb.png"), cv2.cvtColor(detail_rgb, cv2.COLOR_RGB2BGR))

    sparse = np.zeros((3, 40, 40), dtype=np.float32)
    grid_x = int(np.clip(centre[0] / width * 40, 2, 37))
    grid_y = int(np.clip(centre[1] / height * 40, 2, 37))
    sparse[:, grid_y - 2 : grid_y + 2, grid_x - 2 : grid_x + 2] += detail_feature
    sparse_rgb = normalize_uint8(np.transpose(sparse, (1, 2, 0)))
    sparse_rgb = cv2.resize(sparse_rgb, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "08_sparse_detail_map_p4.png"), cv2.cvtColor(sparse_rgb, cv2.COLOR_RGB2BGR))

    p4_map = heatmap(np.mean(np.abs(p4_activation), axis=0), (400, 400))
    fused_map = heatmap(np.mean(np.abs(fused_activation), axis=0), (400, 400))
    cv2.imwrite(str(output_dir / "09_p4_activation_mean.png"), p4_map)
    cv2.imwrite(str(output_dir / "10_fused_feature_mean.png"), fused_map)

    final_image = original_bgr.copy()
    for box in final_boxes:
        xyxy = tuple(int(round(value)) for value in box[:4])
        confidence = float(box[4])
        cv2.rectangle(final_image, xyxy[:2], xyxy[2:], (30, 180, 30), 5)
        cv2.putText(final_image, f"Broken glass {confidence:.3f}", (xyxy[0], max(28, xyxy[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 180, 30), 2)
    cv2.imwrite(str(output_dir / "11_final_detection.jpg"), final_image)

    overview_panels = [
        add_title(original_bgr, "(a) Input image"),
        add_title(annotated, "(b) Candidate and detail crop"),
        add_title(restricted_overlay, "(c) Candidate-restricted Grad-CAM"),
        add_title(crop_preview, "(d) 64x64 detail crop"),
        add_title(detail_channels, "(e) Detail feature: 3x4x4"),
        add_title(sparse_rgb[:, :, ::-1], "(f) Sparse P4 detail map"),
        add_title(p4_map, "(g) P4 classification feature"),
        add_title(fused_map, "(h) Detail-fused feature"),
    ]
    overview = np.concatenate(
        [np.concatenate(overview_panels[:4], axis=1), np.concatenate(overview_panels[4:], axis=1)],
        axis=0,
    )
    cv2.imwrite(str(output_dir / "12_framework_visual_overview.png"), overview)


def main() -> None:
    args = parse_args()
    for path in (args.weight, args.image, args.label, args.archive):
        if not path.exists():
            raise FileNotFoundError(path)

    original_bgr = cv2.imread(str(args.image))
    if original_bgr is None:
        raise RuntimeError(f"Could not read {args.image}")
    image_height, image_width = original_bgr.shape[:2]
    ground_truth_boxes = read_ground_truth(args.label, image_width, image_height)
    padded_bgr, gain, pad_width, pad_height, resized_size = letterbox(original_bgr, args.imgsz)
    input_tensor = torch.from_numpy(cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    captured: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="bgd_p4_visual_") as temporary_directory:
        legacy_val = install_legacy_modules(
            args.archive,
            temporary_directory,
            use_fusion_alpha=True,
            use_multi_head_direct=False,
        )
        model = load_joint_checkpoint(SimpleNamespace(checkpoint=args.weight))
        detection_model = model.model.to(args.device).eval()
        wrapper = detection_model.bgd_318_original_predict_once.__self__
        wrapper.model = detection_model
        wrapper.detail_model = detection_model.detail_model

        batch = {
            "im_file": [str(args.image)],
            "ori_shape": [(image_height, image_width)],
            "resized_shape": [(args.imgsz, args.imgsz)],
            "ratio_pad": [((gain, gain), (pad_width, pad_height))],
        }
        detection_model.batch = batch
        wrapper.batch = batch

        predict_method = detection_model.bgd_318_original_predict_once
        predict_globals = predict_method.__func__.__globals__
        original_find_centres = predict_globals["find_max_heatmap_center_torch"]

        def capture_centres(cam, boxes, window_size):
            result = original_find_centres(cam, boxes, window_size)
            captured["candidate_boxes"] = [item.detach().cpu() for item in boxes]
            captured["centres"] = [item.detach().cpu() for item in result[0]]
            captured["window_size"] = window_size.detach().cpu()
            return result

        predict_globals["find_max_heatmap_center_torch"] = capture_centres
        original_get_cam = wrapper.get_cam

        def capture_cam(output):
            result = original_get_cam(output)
            captured["cam"] = None if result is None else np.asarray(result).copy()
            if wrapper.activations:
                captured["p4"] = wrapper.activations[0].detach().cpu()
            return result

        wrapper.get_cam = capture_cam
        original_get_crops = wrapper.get_crop_images

        def capture_crops(centres, ratio):
            result = original_get_crops(centres, ratio)
            captured["crops"] = result[0].detach().cpu()
            return result

        wrapper.get_crop_images = capture_crops

        detail_handle = wrapper.detail_model.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("detail", output.detach().cpu())
        )
        fused_handle = wrapper.detail_model.relu_for_yolo.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("fused", output.detach().cpu())
        )
        try:
            prediction = detection_model(input_tensor.to(args.device))
        finally:
            detail_handle.remove()
            fused_handle.remove()
            predict_globals["find_max_heatmap_center_torch"] = original_find_centres
            wrapper.get_cam = original_get_cam
            wrapper.get_crop_images = original_get_crops

    required = ("candidate_boxes", "centres", "crops", "cam", "detail", "p4", "fused")
    missing = [name for name in required if captured.get(name) is None]
    if missing:
        raise RuntimeError(f"Missing captured tensors: {missing}")

    candidate_boxes_letterbox = captured["candidate_boxes"][0].numpy()
    centres_letterbox = captured["centres"][0].numpy()
    candidate_boxes_original = np.stack(
        [to_original_box(box, gain, pad_width, pad_height) for box in candidate_boxes_letterbox]
    )
    candidate_boxes_original[:, (0, 2)] = candidate_boxes_original[:, (0, 2)].clip(0, image_width - 1)
    candidate_boxes_original[:, (1, 3)] = candidate_boxes_original[:, (1, 3)].clip(0, image_height - 1)

    overlaps = np.stack(
        [box_iou_one_to_many(gt, candidate_boxes_original) for gt in ground_truth_boxes]
    )
    ground_truth_index, candidate_index = np.unravel_index(int(overlaps.argmax()), overlaps.shape)
    selected_iou = float(overlaps[ground_truth_index, candidate_index])
    selected_candidate = candidate_boxes_original[candidate_index]
    selected_centre = to_original_point(
        centres_letterbox[candidate_index], gain, pad_width, pad_height
    )

    crops = captured["crops"]
    crop = crops[candidate_index]
    mean = crop.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = crop.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    crop_rgb = ((crop * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    detail_feature = captured["detail"][candidate_index].float().numpy()
    cam = np.asarray(captured["cam"])[0]

    from ultralytics.yolo.utils.ops import non_max_suppression

    decoded = prediction[0] if isinstance(prediction, tuple) else prediction
    final = non_max_suppression(decoded.detach(), conf_thres=0.5, iou_thres=0.5)[0].cpu().numpy()
    if len(final):
        final[:, :4] = np.stack(
            [to_original_box(box, gain, pad_width, pad_height) for box in final[:, :4]]
        )
        final[:, (0, 2)] = final[:, (0, 2)].clip(0, image_width - 1)
        final[:, (1, 3)] = final[:, (1, 3)].clip(0, image_height - 1)

    save_visuals(
        args.output,
        original_bgr,
        ground_truth_boxes[ground_truth_index],
        selected_candidate,
        selected_centre,
        crop_rgb,
        cam,
        detail_feature,
        captured["p4"][0].float().numpy(),
        captured["fused"][0].float().numpy(),
        final,
        gain,
        pad_width,
        pad_height,
        resized_size,
    )

    metadata = {
        "source_image": str(args.image),
        "source_label": str(args.label),
        "checkpoint": str(args.weight),
        "selected_ground_truth_xyxy": ground_truth_boxes[ground_truth_index].tolist(),
        "selected_candidate_xyxy": selected_candidate.tolist(),
        "selected_candidate_iou": selected_iou,
        "selected_gradcam_centre_xy": selected_centre.tolist(),
        "candidate_count": int(len(candidate_boxes_original)),
        "final_detection_count_conf_0_5_iou_0_5": int(len(final)),
        "candidate_conf": float(wrapper.conf_threshold),
        "head": "P4/mid",
        "hook": "Detect.cv3[1][1]",
        "detail_crop_shape": list(crop.shape),
        "detail_feature_shape": list(detail_feature.shape),
        "p4_activation_shape": list(captured["p4"][0].shape),
        "fused_activation_shape": list(captured["fused"][0].shape),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FRAMEWORK_VISUAL_OUTPUT", args.output)
    print("FRAMEWORK_VISUAL_METADATA", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
