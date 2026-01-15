from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def match_detections(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    matched_gt = set()
    matched_pred = set()
    matched_ious: List[float] = []
    for p_idx, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou_val = _iou(gt, pred)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            matched_ious.append(best_iou)
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


def summarize_metrics(
    gt_boxes_list: List[List[List[float]]],
    pred_boxes_list: List[List[List[float]]],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    total_tp = total_fp = total_fn = 0
    all_ious: List[float] = []
    count_errors: List[float] = []
    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        tp, fp, fn, matched_ious = match_detections(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(matched_ious)
        count_errors.append(abs(len(pred_boxes) - len(gt_boxes)))
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    count_error = float(np.mean(count_errors)) if count_errors else 0.0
    return precision, recall, mean_iou, count_error
