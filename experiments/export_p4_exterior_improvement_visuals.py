"""Export a complete exterior-glass P4 visual case, including every detail crop."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from export_p4_framework_visuals import (
    box_iou_one_to_many,
    heatmap,
    letterbox,
    normalize_uint8,
    read_ground_truth,
    to_original_box,
    to_original_point,
)
from run_318_fusion_alpha import install_legacy_modules, load_joint_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHOD_WEIGHT = (
    ROOT
    / "experiments/runs/best318_mid_p4_detectconv_learnable_alpha_init05_e50_lr1e4_r2/weights/best.pt"
)
DEFAULT_IMAGE = Path("/home/user/projects/czy/mydata/images/test/00587.jpg")
DEFAULT_LABEL = Path("/home/user/projects/czy/mydata/labels/test/00587.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-weight", type=Path, default=DEFAULT_METHOD_WEIGHT)
    parser.add_argument("--baseline-weight", type=Path, default=ROOT / "best318.pt")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--archive", type=Path, default=ROOT / "what is code.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--candidate-conf", type=float, default=0.2)
    parser.add_argument("--final-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def prepare_input(image_bgr: np.ndarray, size: int):
    padded, gain, pad_width, pad_height, resized_size = letterbox(image_bgr, size)
    tensor = (
        torch.from_numpy(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )
    return tensor, gain, pad_width, pad_height, resized_size


def make_batch(
    image_path: Path,
    height: int,
    width: int,
    size: int,
    gain: float,
    pad_width: float,
    pad_height: float,
):
    return {
        "im_file": [str(image_path)],
        "ori_shape": [(height, width)],
        "resized_shape": [(size, size)],
        "ratio_pad": [((gain, gain), (pad_width, pad_height))],
    }


def convert_predictions(
    prediction: torch.Tensor,
    conf: float,
    iou: float,
    gain: float,
    pad_width: float,
    pad_height: float,
    width: int,
    height: int,
) -> np.ndarray:
    from ultralytics.yolo.utils.ops import non_max_suppression

    result = non_max_suppression(
        prediction.detach(), conf_thres=conf, iou_thres=iou
    )[0].detach().cpu().numpy()
    if len(result):
        result[:, :4] = np.stack(
            [to_original_box(box, gain, pad_width, pad_height) for box in result[:, :4]]
        )
        result[:, (0, 2)] = result[:, (0, 2)].clip(0, width - 1)
        result[:, (1, 3)] = result[:, (1, 3)].clip(0, height - 1)
    return result


def run_baseline(
    args: argparse.Namespace,
    input_tensor: torch.Tensor,
    batch: dict,
    gain: float,
    pad_width: float,
    pad_height: float,
    width: int,
    height: int,
) -> np.ndarray:
    from eval_best318_legacy import install_compatibility_shims, patch_legacy_detail_forward

    install_compatibility_shims()
    with tempfile.TemporaryDirectory(prefix="best318_exterior_baseline_") as directory:
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(directory)
        legacy_root = Path(directory) / "what is code"
        sys.path.insert(0, str(legacy_root))
        sys.path.insert(1, str(ROOT))
        for module_name in ("val", "Resnet6", "gradcam"):
            sys.modules.pop(module_name, None)
        detail_module = importlib.import_module("Resnet6")
        importlib.import_module("gradcam")
        legacy_val = importlib.import_module("val")
        patch_legacy_detail_forward(detail_module)

        checkpoint = torch.load(args.baseline_weight, map_location="cpu", weights_only=False)
        model = checkpoint["model"].float().to(args.device).eval()
        if not isinstance(model.args, dict):
            model.args = vars(model.args).copy()
        wrapper = model._predict_once.__self__
        wrapper.model = model
        wrapper.detail_model = model.detail_model
        for method_name in (
            "_predict_once", "preprocess", "val", "set_requires_grad", "set_hook",
            "save_activation", "save_gradient", "get_cam", "compute_cam_per_layer",
            "aggregate_multi_layers", "get_cam_image", "get_cam_weights",
        ):
            if hasattr(legacy_val.BGD_YOLO, method_name):
                setattr(
                    wrapper,
                    method_name,
                    types.MethodType(getattr(legacy_val.BGD_YOLO, method_name), wrapper),
                )
        wrapper.conf_threshold = args.candidate_conf
        wrapper.iou_threshold = args.iou
        wrapper.validator = None
        wrapper.batch = batch
        model.batch = batch
        original_forward = wrapper._predict_once

        def grad_enabled_forward(_model, images, profile=False, visualize=False):
            with torch.inference_mode(False), torch.enable_grad():
                images = images.detach().clone().requires_grad_(True)
                return original_forward(images, profile, visualize)

        model._predict_once = types.MethodType(grad_enabled_forward, model)
        output = model(input_tensor.to(args.device))
        decoded = output[0] if isinstance(output, tuple) else output
        final = convert_predictions(
            decoded, args.final_conf, args.iou, gain, pad_width, pad_height, width, height
        )
        del output, decoded, checkpoint, wrapper, model
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return final


def run_method(
    args: argparse.Namespace,
    input_tensor: torch.Tensor,
    batch: dict,
) -> tuple[np.ndarray, dict[str, object], dict]:
    captured: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="bgd_p4_exterior_") as directory:
        install_legacy_modules(
            args.archive,
            directory,
            use_fusion_alpha=True,
            use_multi_head_direct=False,
        )
        yolo = load_joint_checkpoint(SimpleNamespace(checkpoint=args.method_weight))
        model = yolo.model.to(args.device).float().eval()
        wrapper = model.bgd_318_original_predict_once.__self__
        wrapper.model = model
        wrapper.detail_model = model.detail_model
        if abs(float(wrapper.conf_threshold) - args.candidate_conf) > 1e-9:
            raise RuntimeError(
                f"Checkpoint candidate confidence is {wrapper.conf_threshold}, "
                f"not requested {args.candidate_conf}"
            )
        wrapper.batch = batch
        model.batch = batch

        predict_globals = model.bgd_318_original_predict_once.__func__.__globals__
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
            decoded_before_fusion = output[0]
            condition = decoded_before_fusion[0, -1] > wrapper.conf_threshold
            selected = decoded_before_fusion[0, :, condition].T
            captured["candidate_scores"] = selected[:, -1].detach().cpu()
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
            lambda _module, _inputs, output: captured.__setitem__(
                "detail", output.detach().cpu()
            )
        )
        fused_handle = wrapper.detail_model.relu_for_yolo.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "fused", output.detach().cpu()
            )
        )
        try:
            output = model(input_tensor.to(args.device))
        finally:
            detail_handle.remove()
            fused_handle.remove()
            predict_globals["find_max_heatmap_center_torch"] = original_find_centres
            wrapper.get_cam = original_get_cam
            wrapper.get_crop_images = original_get_crops

        decoded = output[0] if isinstance(output, tuple) else output
        config = dict(model.bgd_318_alpha_config)
        alpha_parameter = getattr(model, "fusion_alpha_logit_bias", None)
        if alpha_parameter is not None:
            config["fusion_alpha_current"] = float(
                torch.sigmoid(alpha_parameter.detach()).cpu()
            )
        prediction = decoded.detach().cpu()
    return prediction, captured, config


def match_metrics(gt: np.ndarray, boxes: np.ndarray) -> dict:
    if not len(gt) or not len(boxes):
        return {"best_iou": 0.0, "matched_confidence": 0.0}
    overlaps = np.stack([box_iou_one_to_many(item, boxes[:, :4]) for item in gt])
    gt_index, prediction_index = np.unravel_index(int(overlaps.argmax()), overlaps.shape)
    return {
        "best_iou": float(overlaps[gt_index, prediction_index]),
        "matched_confidence": float(boxes[prediction_index, 4]),
        "ground_truth_index": int(gt_index),
        "prediction_index": int(prediction_index),
    }


def detection_counts(gt: np.ndarray, boxes: np.ndarray, threshold: float = 0.5) -> dict:
    """Greedily assign predictions to GT so TP/FP corrections are explicit."""
    assignments = ["FP"] * len(boxes)
    if len(gt) and len(boxes):
        overlaps = np.stack([box_iou_one_to_many(item, boxes[:, :4]) for item in gt])
        available_gt = set(range(len(gt)))
        available_predictions = set(range(len(boxes)))
        while available_gt and available_predictions:
            best = max(
                (
                    (float(overlaps[gt_index, prediction_index]), gt_index, prediction_index)
                    for gt_index in available_gt
                    for prediction_index in available_predictions
                ),
                default=(0.0, -1, -1),
            )
            if best[0] < threshold:
                break
            _, gt_index, prediction_index = best
            assignments[prediction_index] = "TP"
            available_gt.remove(gt_index)
            available_predictions.remove(prediction_index)
    tp = assignments.count("TP")
    return {
        "tp": tp,
        "fp": len(boxes) - tp,
        "fn": len(gt) - tp,
        "prediction_assignments": assignments,
    }


def annotation_scale(image: np.ndarray) -> float:
    """Keep lines and labels bold after high-resolution figures are downscaled."""
    return max(1.0, min(image.shape[:2]) / 1200.0)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int],
    prefix: str,
    thickness: int = 9,
) -> np.ndarray:
    result = image.copy()
    scale = annotation_scale(image)
    line_width = max(5, int(round(thickness * scale)))
    for index, box in enumerate(boxes):
        xyxy = tuple(int(round(value)) for value in box[:4])
        label = prefix
        if len(box) > 4:
            label = f"{prefix} {float(box[4]):.3f}"
        cv2.rectangle(result, xyxy[:2], xyxy[2:], color, line_width)
        cv2.putText(
            result,
            label,
            (xyxy[0], max(30, xyxy[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.12 * scale,
            color,
            max(3, int(round(4 * scale))),
        )
    return result


def draw_detection_result(
    image: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Draw thick GT halos plus green TP and red FP boxes."""
    result = image.copy()
    scale = annotation_scale(image)
    gt_width = max(10, int(round(15 * scale)))
    prediction_width = max(6, int(round(8 * scale)))
    font_scale = 1.15 * scale
    font_width = max(3, int(round(4 * scale)))
    counts = detection_counts(ground_truth, predictions, iou_threshold)

    for index, box in enumerate(ground_truth, 1):
        xyxy = tuple(int(round(value)) for value in box[:4])
        cv2.rectangle(result, xyxy[:2], xyxy[2:], (0, 220, 255), gt_width)
        cv2.putText(
            result,
            f"GT {index}",
            (xyxy[0], min(image.shape[0] - 18, xyxy[1] + int(round(52 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 170, 210),
            font_width,
        )

    for index, (box, assignment) in enumerate(
        zip(predictions, counts["prediction_assignments"]), 1
    ):
        xyxy = tuple(int(round(value)) for value in box[:4])
        # Use a single vivid red for prediction boxes so the detection result
        # remains unmistakable after high-resolution images are downscaled.
        color = (0, 0, 255)
        cv2.rectangle(result, xyxy[:2], xyxy[2:], color, prediction_width)
        cv2.putText(
            result,
            f"{assignment} {float(box[4]):.3f}",
            (xyxy[0], max(int(round(42 * scale)), xyxy[1] - int(round(18 * scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_width,
        )
    return result, counts


def title_panel(image: np.ndarray, title: str, subtitle: str, width: int = 660) -> np.ndarray:
    header = 112
    scale = width / image.shape[1]
    resized = cv2.resize(
        image,
        (width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((resized.shape[0] + header, width, 3), 255, dtype=np.uint8)
    canvas[header:] = resized
    cv2.putText(canvas, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (25, 25, 25), 3)
    cv2.putText(canvas, subtitle, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (65, 65, 65), 2)
    return canvas


def crop_to_rgb(crop: torch.Tensor) -> np.ndarray:
    mean = crop.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = crop.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (
        ((crop * std + mean).clamp(0, 1) * 255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )


def main() -> None:
    args = parse_args()
    if args.device.isdigit():
        args.device = f"cuda:{args.device}"
    for path in (
        args.method_weight, args.baseline_weight, args.image, args.label, args.archive
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    original = cv2.imread(str(args.image))
    if original is None:
        raise RuntimeError(f"Could not read {args.image}")
    height, width = original.shape[:2]
    ground_truth = read_ground_truth(args.label, width, height)
    input_tensor, gain, pad_width, pad_height, resized_size = prepare_input(
        original, args.imgsz
    )
    batch = make_batch(
        args.image, height, width, args.imgsz, gain, pad_width, pad_height
    )

    method_prediction, captured, method_config = run_method(args, input_tensor, batch)
    method = convert_predictions(
        method_prediction,
        args.final_conf,
        args.iou,
        gain,
        pad_width,
        pad_height,
        width,
        height,
    )
    baseline = run_baseline(
        args, input_tensor, batch, gain, pad_width, pad_height, width, height
    )

    required = (
        "candidate_boxes", "candidate_scores", "centres", "crops", "cam",
        "detail", "p4", "fused",
    )
    missing = [key for key in required if captured.get(key) is None]
    if missing:
        raise RuntimeError(f"Missing captured tensors: {missing}")

    candidate_boxes_letterbox = captured["candidate_boxes"][0].numpy()
    candidate_scores = captured["candidate_scores"].numpy()
    if len(candidate_boxes_letterbox) != len(candidate_scores):
        raise RuntimeError("Candidate box/score count mismatch")
    candidate_boxes = np.stack(
        [
            to_original_box(box, gain, pad_width, pad_height)
            for box in candidate_boxes_letterbox
        ]
    )
    candidate_boxes[:, (0, 2)] = candidate_boxes[:, (0, 2)].clip(0, width - 1)
    candidate_boxes[:, (1, 3)] = candidate_boxes[:, (1, 3)].clip(0, height - 1)
    candidate_rows = np.concatenate([candidate_boxes, candidate_scores[:, None]], axis=1)
    centres = np.stack(
        [
            to_original_point(point, gain, pad_width, pad_height)
            for point in captured["centres"][0].numpy()
        ]
    )
    if len(candidate_rows) != len(centres):
        raise RuntimeError("Candidate/Grad-CAM centre count mismatch")

    output = args.output
    crop_dir = output / "candidate_crops_64"
    crop_preview_dir = output / "candidate_crops_preview"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "01_exterior_input.jpg"), original)
    gt_image = draw_boxes(original, ground_truth, (40, 185, 40), "Ground truth")
    cv2.imwrite(str(output / "02_ground_truth.jpg"), gt_image)

    candidate_image = original.copy()
    figure_scale = annotation_scale(original)
    candidate_line_width = max(7, int(round(9 * figure_scale)))
    candidate_font_scale = 1.08 * figure_scale
    candidate_font_width = max(3, int(round(4 * figure_scale)))
    palette = [
        (26, 46, 220), (230, 90, 25), (180, 35, 180), (30, 170, 220),
        (200, 120, 20), (30, 160, 90), (150, 80, 210), (80, 80, 230),
    ]
    for index, (box, centre) in enumerate(zip(candidate_rows, centres), 1):
        color = palette[(index - 1) % len(palette)]
        xyxy = tuple(int(round(value)) for value in box[:4])
        cv2.rectangle(
            candidate_image, xyxy[:2], xyxy[2:], color, candidate_line_width
        )
        cv2.circle(
            candidate_image,
            tuple(int(round(v)) for v in centre),
            max(11, int(round(14 * figure_scale))),
            (0, 235, 255),
            -1,
        )
        cv2.putText(
            candidate_image,
            f"C{index:02d}",
            (
                xyxy[0] + int(round(24 * figure_scale)),
                max(
                    int(round(42 * figure_scale)),
                    xyxy[1] - int(round((22 + (index - 1) * 52) * figure_scale)),
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            candidate_font_scale,
            color,
            candidate_font_width,
        )
    legend_width = max(700, int(round(760 * figure_scale)))
    candidate_with_legend = np.full(
        (height, width + legend_width, 3), 255, dtype=np.uint8
    )
    candidate_with_legend[:, :width] = candidate_image
    cv2.putText(
        candidate_with_legend,
        f"All {len(candidate_rows)} raw candidates: conf > {args.candidate_conf}",
        (width + 24, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.00 * figure_scale,
        (25, 25, 25),
        max(3, int(round(4 * figure_scale))),
    )
    for index, box in enumerate(candidate_rows, 1):
        color = palette[(index - 1) % len(palette)]
        y = int(round((120 + (index - 1) * 104) * figure_scale))
        marker = max(30, int(round(36 * figure_scale)))
        cv2.rectangle(
            candidate_with_legend,
            (width + 28, y - marker),
            (width + 28 + marker, y),
            color,
            -1,
        )
        cv2.putText(
            candidate_with_legend,
            f"C{index:02d}  conf={box[4]:.4f}",
            (width + 82, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88 * figure_scale,
            (30, 30, 30),
            max(3, int(round(3 * figure_scale))),
        )
        cv2.putText(
            candidate_with_legend,
            f"xyxy=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})",
            (width + 82, y + int(round(42 * figure_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * figure_scale,
            (70, 70, 70),
            max(2, int(round(2 * figure_scale))),
        )
    cv2.putText(
        candidate_with_legend,
        "Overlapping anchors share the same Grad-CAM crop center.",
        (
            width + 24,
            min(
                height - 50,
                int(round((120 + len(candidate_rows) * 104 + 55) * figure_scale)),
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * figure_scale,
        (70, 70, 70),
        max(2, int(round(2 * figure_scale))),
    )
    cv2.imwrite(
        str(output / "03_all_candidates_conf_gt_0_2.jpg"), candidate_with_legend
    )

    resized_width, resized_height = resized_size
    left, top = int(round(pad_width - 0.1)), int(round(pad_height - 0.1))
    cam_letterbox = np.asarray(captured["cam"])[0]
    cam_unpadded = cam_letterbox[top : top + resized_height, left : left + resized_width]
    cam_original = cv2.resize(cam_unpadded, (width, height), interpolation=cv2.INTER_LINEAR)
    colored_cam = cv2.applyColorMap(normalize_uint8(cam_original), cv2.COLORMAP_JET)
    cam_overlay = cv2.addWeighted(original, 0.52, colored_cam, 0.48, 0)
    for index, (box, centre) in enumerate(zip(candidate_rows, centres), 1):
        xyxy = tuple(int(round(value)) for value in box[:4])
        cv2.rectangle(
            cam_overlay, xyxy[:2], xyxy[2:], (255, 255, 255), candidate_line_width
        )
        point = tuple(int(round(value)) for value in centre)
        cv2.circle(
            cam_overlay,
            point,
            max(11, int(round(14 * figure_scale))),
            (0, 255, 255),
            -1,
        )
        cv2.putText(
            cam_overlay,
            f"C{index:02d}",
            (
                point[0] + int(round(24 * figure_scale)),
                max(
                    int(round(42 * figure_scale)),
                    point[1] - int(round((18 + (index - 1) * 52) * figure_scale)),
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            candidate_font_scale,
            (0, 0, 0),
            candidate_font_width,
        )
    cv2.imwrite(str(output / "04_p4_gradcam_all_candidates.jpg"), cam_overlay)
    cv2.imwrite(str(output / "05_p4_gradcam_heatmap.png"), normalize_uint8(cam_original))

    crop_panels = []
    crop_metadata = []
    crops = captured["crops"]
    for index, (crop, box, centre) in enumerate(zip(crops, candidate_rows, centres), 1):
        rgb = crop_to_rgb(crop)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        crop_name = f"C{index:02d}_conf_{box[4]:.4f}.png"
        cv2.imwrite(str(crop_dir / crop_name), bgr)
        preview = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(crop_preview_dir / crop_name), preview)
        panel = np.full((372, 340, 3), 255, dtype=np.uint8)
        panel[42:362, 10:330] = preview
        cv2.putText(
            panel,
            f"C{index:02d}  conf={box[4]:.4f}",
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (30, 30, 30),
            2,
        )
        crop_panels.append(panel)
        crop_metadata.append(
            {
                "id": f"C{index:02d}",
                "confidence": float(box[4]),
                "box_xyxy": box[:4].tolist(),
                "gradcam_center_xy": centre.tolist(),
                "crop_64": f"candidate_crops_64/{crop_name}",
                "crop_preview": f"candidate_crops_preview/{crop_name}",
            }
        )
    columns = min(4, len(crop_panels))
    rows = []
    for start in range(0, len(crop_panels), columns):
        group = crop_panels[start : start + columns]
        while len(group) < columns:
            group.append(np.full_like(crop_panels[0], 255))
        rows.append(np.concatenate(group, axis=1))
    crop_montage = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output / "06_all_candidate_crops_montage.png"), crop_montage)

    detail_features = captured["detail"].float().numpy()
    feature_panels = []
    for index, feature in enumerate(detail_features, 1):
        rgb = normalize_uint8(np.transpose(feature, (1, 2, 0)))
        rgb = cv2.resize(rgb, (240, 240), interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(panel, f"C{index:02d}  3x4x4", (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        feature_panels.append(panel)
    cv2.imwrite(
        str(output / "07_all_detail_features.png"), np.concatenate(feature_panels, axis=1)
    )

    p4_map = heatmap(np.mean(np.abs(captured["p4"][0].float().numpy()), axis=0), (400, 400))
    fused_map = heatmap(
        np.mean(np.abs(captured["fused"][0].float().numpy()), axis=0), (400, 400)
    )
    cv2.imwrite(str(output / "08_p4_activation_mean.png"), p4_map)
    cv2.imwrite(str(output / "09_fused_activation_mean.png"), fused_map)

    baseline_image, baseline_counts = draw_detection_result(
        original, ground_truth, baseline, args.iou
    )
    method_image, method_counts = draw_detection_result(
        original, ground_truth, method, args.iou
    )
    cv2.imwrite(str(output / "10_baseline_best318_final.jpg"), baseline_image)
    cv2.imwrite(str(output / "11_ours_p4_final.jpg"), method_image)
    baseline_metrics = match_metrics(ground_truth, baseline)
    method_metrics = match_metrics(ground_truth, method)
    baseline_panel = title_panel(
        baseline_image,
        "Baseline: best318.pt",
        (
            f"TP={baseline_counts['tp']} FP={baseline_counts['fp']} FN={baseline_counts['fn']}  "
            f"IoU={baseline_metrics['best_iou']:.3f}"
        ),
    )
    method_panel = title_panel(
        method_image,
        "Ours: P4 Grad-CAM + Detail Fusion",
        (
            f"TP={method_counts['tp']} FP={method_counts['fp']} FN={method_counts['fn']}  "
            f"IoU={method_metrics['best_iou']:.3f}"
        ),
    )
    comparison = np.concatenate([baseline_panel, method_panel], axis=1)
    cv2.imwrite(str(output / "12_baseline_vs_ours_comparison.jpg"), comparison)

    flow_panels = [
        title_panel(original, "Input", "Exterior broken window", 360),
        title_panel(candidate_image, "Candidates", f"all {len(candidate_rows)} boxes: conf>0.2", 360),
        title_panel(cam_overlay, "P4 Grad-CAM", "yellow dots: crop centers", 360),
        title_panel(
            method_image,
            "Final result",
            f"TP={method_counts['tp']} FP={method_counts['fp']} conf={method_metrics['matched_confidence']:.3f}",
            360,
        ),
    ]
    minimum_height = min(panel.shape[0] for panel in flow_panels)
    flow_panels = [panel[:minimum_height] for panel in flow_panels]
    cv2.imwrite(str(output / "13_exterior_framework_flow.jpg"), np.concatenate(flow_panels, axis=1))

    metadata = {
        "source_image": str(args.image),
        "source_label": str(args.label),
        "baseline_checkpoint": str(args.baseline_weight),
        "method_checkpoint": str(args.method_weight),
        "candidate_conf_strictly_greater_than": args.candidate_conf,
        "final_conf": args.final_conf,
        "nms_iou": args.iou,
        "candidate_count": len(candidate_rows),
        "candidates": crop_metadata,
        "gradcam": {
            "hook": "Detect.cv3[1][1]",
            "shape": list(cam_original.shape),
            "minimum": float(cam_original.min()),
            "maximum": float(cam_original.max()),
            "sum": float(cam_original.sum()),
            "nonzero_pixels": int(np.count_nonzero(cam_original)),
        },
        "ground_truth_boxes_xyxy": ground_truth.tolist(),
        "baseline_final_boxes": baseline.tolist(),
        "method_final_boxes": method.tolist(),
        "baseline_metrics": baseline_metrics,
        "method_metrics": method_metrics,
        "baseline_detection_counts": baseline_counts,
        "method_detection_counts": method_counts,
        "improvement": {
            "confidence_absolute": method_metrics["matched_confidence"]
            - baseline_metrics["matched_confidence"],
            "iou_absolute": method_metrics["best_iou"] - baseline_metrics["best_iou"],
        },
        "method_config": method_config,
        "tensor_shapes": {
            "detail_crops": list(crops.shape),
            "detail_features": list(captured["detail"].shape),
            "p4_activation": list(captured["p4"][0].shape),
            "fused_activation": list(captured["fused"][0].shape),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTERIOR_VISUAL_OUTPUT", output)
    print("EXTERIOR_VISUAL_METADATA", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
