"""Minimal helpers required by the historical BGD wrapper."""

import numpy as np
import torch

from ultralytics.yolo.utils.ops import xywh2xyxy


def convert_to_original_coords(x_padded, y_padded, ratio, dw, dh):
    """Map a point from the letterboxed input back to its original image."""
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    return (x_padded - left) / ratio, (y_padded - top) / ratio


def find_max_heatmap_center_torch(heatmap, boxes, window_size=64):
    """Historical all-ones window search retained for wrapper compatibility."""
    batch_size, height, width = heatmap.shape
    device = boxes[0].device
    heatmap = torch.as_tensor(heatmap, dtype=torch.float32, device=device)
    centers, candidate_to_box = [], []

    for batch_index in range(batch_size):
        batch_boxes = boxes[batch_index].detach().clone().long()
        window = int(window_size[batch_index].item())
        convolution = torch.nn.Conv2d(1, 1, window, bias=False).to(device)
        convolution.weight.data.fill_(1.0)
        convolution.requires_grad_(False)
        with torch.no_grad():
            response = convolution(heatmap[batch_index][None, None])[0, 0]

        batch_boxes[:, [0, 2]] = batch_boxes[:, [0, 2]].clamp(0, width)
        batch_boxes[:, [1, 3]] = batch_boxes[:, [1, 3]].clamp(0, height)
        valid = ((batch_boxes[:, 2] - batch_boxes[:, 0]) >= window) & (
            (batch_boxes[:, 3] - batch_boxes[:, 1]) >= window
        )
        batch_centers = torch.zeros((len(batch_boxes), 2), dtype=torch.long, device=device)

        invalid = batch_boxes[~valid]
        if len(invalid):
            batch_centers[~valid] = torch.stack((
                (invalid[:, 0] + invalid[:, 2]) // 2,
                (invalid[:, 1] + invalid[:, 3]) // 2,
            ), dim=1)

        valid_boxes = batch_boxes[valid]
        if len(valid_boxes):
            response_boxes = valid_boxes.clone()
            response_boxes[:, 2] = (response_boxes[:, 2] - window // 2 + 1).clamp(min=0)
            response_boxes[:, 3] = (response_boxes[:, 3] - window // 2 + 1).clamp(min=0)
            maxima = []
            for x1, y1, x2, y2 in response_boxes:
                region = response[y1:y2, x1:x2]
                if not region.numel():
                    maxima.append([-1, -1])
                    continue
                max_y, max_x = np.unravel_index(int(region.argmax()), region.shape)
                maxima.append([max_x + x1, max_y + y1])
            maxima = torch.tensor(maxima, device=device)
            fallback = (maxima == -1).any(dim=1)
            valid_centers = maxima + window // 2
            if fallback.any():
                fallback_boxes = valid_boxes[fallback]
                valid_centers[fallback] = torch.stack((
                    (fallback_boxes[:, 0] + fallback_boxes[:, 2]) // 2,
                    (fallback_boxes[:, 1] + fallback_boxes[:, 3]) // 2,
                ), dim=1)
            batch_centers[valid.nonzero(as_tuple=False).squeeze(1)] = valid_centers.long()

        centers.append(batch_centers)
        _, inverse = torch.unique(batch_centers, dim=0, return_inverse=True)
        mapping = [[] for _ in range(int(inverse.max()) + 1)] if inverse.numel() else []
        for box_index, center_index in enumerate(inverse):
            mapping[int(center_index)].append(box_index)
        candidate_to_box.append(mapping)

    return centers, candidate_to_box


class yolov8_target_batch(torch.nn.Module):
    """Historical Grad-CAM score target used by the BGD wrapper."""

    @staticmethod
    def post_process(result):
        logits = result[:, 4:]
        boxes = result[:, :4]
        return logits.transpose(1, 2), boxes.transpose(1, 2), xywh2xyxy(boxes.detach().cpu().numpy())

    def forward(self, data):
        logits, boxes, _ = self.post_process(data)
        return torch.stack([sum(logits[index]) for index in range(boxes.shape[0])])
