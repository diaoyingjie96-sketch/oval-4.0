"""
Oval defect detector for Fab AOI images.

Install dependencies:
    pip install opencv-python-headless numpy pandas tqdm matplotlib

Default Windows layout:
    C:\\defect_ai\\
        bad\\
        good\\
        unknown\\
        output\\

Run:
    python C:\\defect_ai\\detect_oval_defect.py
    python C:\\defect_ai\\detect_oval_defect.py --root C:\\defect_ai

This is a traditional computer-vision pipeline. It detects weak oval shadow
rings/blobs using local, per-image contrast and ellipse-consistency metrics.
It intentionally does not use a gold-color threshold as the primary decision.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Config:
    # Keep this as 482 when production inputs are 482x482. Larger images are
    # downscaled for scale-stable metrics; centers/axes are mapped back.
    process_size: int = 482

    # Pre-processing.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    row_normalize: bool = True
    small_blur_sigma: float = 2.2
    dog_sigmas: Tuple[Tuple[float, float], ...] = ((3.0, 15.0), (5.0, 25.0), (8.0, 42.0))

    # Candidate generation. Fractions are relative to processed image area.
    # Lower percentiles plus larger close kernels recover weak semi-closed
    # oval rings. Later scoring suppresses pad/trace structures.
    response_percentiles: Tuple[float, ...] = (92.0, 94.0, 96.0, 97.2, 98.2, 99.0)
    close_kernel_fracs: Tuple[float, ...] = (0.022, 0.045, 0.073)
    min_area_frac: float = 0.00145
    max_area_frac: float = 0.12
    min_minor_frac: float = 0.025
    max_major_frac: float = 0.72
    max_axis_ratio: float = 5.8
    min_contour_points: int = 12
    max_candidates_per_image: int = 30

    # Scoring and suppression.
    border_margin_frac: float = 0.018
    long_line_kernel_frac: float = 0.095
    structure_dilate_frac: float = 0.012
    green_text_dilate_frac: float = 0.018
    weak_score_floor: float = 0.0

    # Validation thresholding.
    min_bad_recall_preference: float = 0.90
    borderline_margin_abs: float = 5.0
    borderline_margin_rel: float = 0.08
    min_validation_bad: int = 3
    min_validation_good: int = 3


CONFIG = Config()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resize_for_processing(image: np.ndarray, process_size: int) -> Tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]
    if process_size <= 0 or (h == process_size and w == process_size):
        return image.copy(), 1.0, 1.0
    resized = cv2.resize(image, (process_size, process_size), interpolation=cv2.INTER_AREA)
    return resized, w / process_size, h / process_size


def normalize_u8(x: np.ndarray, mask: Optional[np.ndarray] = None, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    xf = x.astype(np.float32)
    values = xf[mask > 0] if mask is not None and np.any(mask > 0) else xf.ravel()
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if hi <= lo + 1e-6:
        return np.zeros_like(xf, dtype=np.uint8)
    y = np.clip((xf - lo) * 255.0 / (hi - lo), 0, 255)
    return y.astype(np.uint8)


def robust_stats(x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    vals = x[mask > 0] if mask is not None and np.any(mask > 0) else x.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(vals) + 1e-6)
    return med, sigma


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def ellipse_mask(shape: Tuple[int, int], ellipse: Tuple[Tuple[float, float], Tuple[float, float], float], scale: float = 1.0, thickness: int = -1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    (cx, cy), (major, minor), angle = ellipse
    axes = (max(1, int(round(major * scale / 2.0))), max(1, int(round(minor * scale / 2.0))))
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), axes, float(angle), 0, 360, 255, thickness)
    return mask


def annulus_mask(shape: Tuple[int, int], ellipse: Tuple[Tuple[float, float], Tuple[float, float], float], inner: float, outer: float) -> np.ndarray:
    out = ellipse_mask(shape, ellipse, outer, -1)
    inn = ellipse_mask(shape, ellipse, inner, -1)
    return cv2.subtract(out, inn)


def green_overlay_mask(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # AOI overlays are bright green text/scale marks. They are not defects.
    mask = cv2.inRange(hsv, np.array([35, 70, 60], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8))
    k = max(3, int(round(min(bgr.shape[:2]) * cfg.green_text_dilate_frac)))
    if k % 2 == 0:
        k += 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def preprocess(bgr: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    green_mask = green_overlay_mask(bgr, cfg)
    valid_mask = (green_mask == 0).astype(np.uint8) * 255

    # Illumination correction removes slow microscope shading without assuming
    # the defect is in any specific color/material region.
    sigma_bg = max(21, int(round(min(l.shape) * 0.085)))
    bg = cv2.GaussianBlur(l, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
    illum = l - bg + 128.0

    # Row-wise normalization suppresses horizontal stripe texture and broad
    # horizontal bands. A local oval remains because it affects only a limited
    # x-range in the same row.
    if cfg.row_normalize:
        row_med = np.median(illum, axis=1, keepdims=True)
        stripe_removed = illum - row_med + np.median(illum)
    else:
        stripe_removed = illum.copy()

    stripe_u8 = normalize_u8(stripe_removed, valid_mask)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid))
    contrast = clahe.apply(stripe_u8)
    smooth = cv2.GaussianBlur(contrast.astype(np.float32), (0, 0), cfg.small_blur_sigma)

    responses = []
    for small_sigma, large_sigma in cfg.dog_sigmas:
        small = cv2.GaussianBlur(smooth, (0, 0), small_sigma)
        large = cv2.GaussianBlur(smooth, (0, 0), large_sigma)
        dark = large - small  # positive means a local dark shadow/blob
        responses.append(np.maximum(dark, 0))
    dark_response = np.max(np.stack(responses, axis=0), axis=0)

    _, r_sigma = robust_stats(dark_response, valid_mask)
    dark_z = dark_response / max(r_sigma, 1e-6)
    dark_z_u8 = normalize_u8(dark_z, valid_mask, 1, 99.6)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = normalize_u8(grad, valid_mask, 1, 99.3)

    structure = structure_mask(grad_u8, green_mask, cfg)

    return {
        "gray": gray,
        "raw_l": l,
        "green_mask": green_mask,
        "valid_mask": valid_mask,
        "illumination_corrected": normalize_u8(illum, valid_mask),
        "contrast_enhanced": contrast,
        "stripe_removed": stripe_u8,
        "dark_response": dark_z,
        "dark_response_u8": dark_z_u8,
        "gradient": grad,
        "gradient_map": grad_u8,
        "gx": gx,
        "gy": gy,
        "structure_mask": structure,
    }


def structure_mask(grad_u8: np.ndarray, green_mask: np.ndarray, cfg: Config) -> np.ndarray:
    h, w = grad_u8.shape
    valid_vals = grad_u8[green_mask == 0]
    thr = max(35, float(np.percentile(valid_vals, 96.5))) if valid_vals.size else 60
    high = (grad_u8 >= thr).astype(np.uint8) * 255
    long_k = max(15, int(round(min(h, w) * cfg.long_line_kernel_frac)))
    dil_k = max(3, int(round(min(h, w) * cfg.structure_dilate_frac)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (long_k, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_k))
    horizontal = cv2.morphologyEx(high, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(high, cv2.MORPH_OPEN, v_kernel)
    mask = cv2.bitwise_or(horizontal, vertical)
    mask = cv2.bitwise_or(mask, green_mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k)))
    return mask


def component_ellipse(contour: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    if contour is None or len(contour) < 5:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (a, b), angle = ellipse
    major = max(a, b)
    minor = min(a, b)
    if major <= 1 or minor <= 1:
        return None
    # Store as (major, minor) while cv2.ellipse drawing uses axes independent
    # of angle for our mask scoring. Angle remains the fit angle.
    return (float(cx), float(cy)), (float(major), float(minor)), float(angle)


def contour_from_component(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def candidate_masks_from_response(pre: Dict[str, np.ndarray], cfg: Config) -> Tuple[np.ndarray, List[np.ndarray]]:
    response = pre["dark_response"]
    valid = pre["valid_mask"]
    structure = pre["structure_mask"]
    h, w = response.shape
    values = response[(valid > 0) & (structure == 0)]
    if values.size < 100:
        values = response[valid > 0]
    masks: List[np.ndarray] = []
    all_mask = np.zeros((h, w), dtype=np.uint8)

    for p in cfg.response_percentiles:
        thr = float(np.percentile(values, p)) if values.size else float(np.percentile(response, p))
        binary = ((response >= thr) & (valid > 0)).astype(np.uint8) * 255
        k1 = max(3, int(round(min(h, w) * 0.010)))
        if k1 % 2 == 0:
            k1 += 1
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1)))
        for close_frac in cfg.close_kernel_fracs:
            k2 = max(5, int(round(min(h, w) * close_frac)))
            if k2 % 2 == 0:
                k2 += 1
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area <= 0:
                    continue
                cm = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(cm, [c], -1, 255, -1)
                masks.append(cm)
                all_mask = cv2.bitwise_or(all_mask, cm)

    return all_mask, masks


def gradient_orientation_anisotropy(gx: np.ndarray, gy: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if int(np.sum(m)) < 20:
        return 1.0
    gxx = gx[m].astype(np.float64)
    gyy = gy[m].astype(np.float64)
    mag = np.sqrt(gxx * gxx + gyy * gyy) + 1e-6
    ang = np.arctan2(gyy, gxx)
    c = np.sum(mag * np.cos(2.0 * ang))
    s = np.sum(mag * np.sin(2.0 * ang))
    return float(np.sqrt(c * c + s * s) / np.sum(mag))


def ellipse_ring_continuity(
    grad: np.ndarray,
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
    outer_grad_ref: float,
    sectors: int = 72,
) -> Tuple[float, float]:
    (cx, cy), (major, minor), angle = ellipse
    a = max(major / 2.0, 1.0)
    b = max(minor / 2.0, 1.0)
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    h, w = grad.shape
    values = []
    for i in range(sectors):
        t0 = 2.0 * math.pi * i / sectors
        sector_vals = []
        for dt in (-0.035, 0.0, 0.035):
            t = t0 + dt
            x = cx + a * math.cos(t) * cos_t - b * math.sin(t) * sin_t
            y = cy + a * math.cos(t) * sin_t + b * math.sin(t) * cos_t
            ix = int(round(x))
            iy = int(round(y))
            if 1 <= ix < w - 1 and 1 <= iy < h - 1:
                sector_vals.append(float(grad[iy, ix]))
        if sector_vals:
            values.append(float(np.mean(sector_vals)))
        else:
            values.append(0.0)
    vals = np.array(values, dtype=np.float32)
    if vals.size == 0:
        return 0.0, 0.0
    threshold = max(float(np.percentile(vals, 45)), outer_grad_ref * 1.10)
    continuity = float(np.mean(vals > threshold))
    strength = float(np.mean(vals) - outer_grad_ref)
    return continuity, strength


def score_candidate(
    comp_mask: np.ndarray,
    pre: Dict[str, np.ndarray],
    cfg: Config,
) -> Optional[Dict[str, float]]:
    h, w = comp_mask.shape
    img_area = float(h * w)
    area = float(np.sum(comp_mask > 0))
    if area < cfg.min_area_frac * img_area or area > cfg.max_area_frac * img_area:
        return None

    contour = contour_from_component(comp_mask)
    if contour is None or len(contour) < cfg.min_contour_points:
        return None
    ellipse = component_ellipse(contour)
    if ellipse is None:
        return None
    (cx, cy), (major, minor), angle = ellipse
    axis_ratio = major / max(minor, 1e-6)
    if axis_ratio > cfg.max_axis_ratio:
        return None
    if minor < cfg.min_minor_frac * min(h, w) or major > cfg.max_major_frac * min(h, w):
        return None

    ellipse_full = ellipse_mask((h, w), ellipse, 1.0, -1)
    ellipse_outer = ellipse_mask((h, w), ellipse, 1.32, -1)
    ellipse_inner = ellipse_mask((h, w), ellipse, 0.62, -1)
    ring = annulus_mask((h, w), ellipse, 0.82, 1.14)
    outer = cv2.subtract(ellipse_outer, ellipse_mask((h, w), ellipse, 1.08, -1))

    if np.sum(ellipse_full > 0) < 20 or np.sum(outer > 0) < 20 or np.sum(ring > 0) < 20:
        return None

    intersection = np.sum((comp_mask > 0) & (ellipse_full > 0))
    union = np.sum((comp_mask > 0) | (ellipse_full > 0))
    ellipse_iou = float(intersection / max(union, 1))
    ring_intersection = np.sum((comp_mask > 0) & (ring > 0))
    ring_union = np.sum((comp_mask > 0) | (ring > 0))
    ring_iou = float(ring_intersection / max(ring_union, 1))
    fit_iou = max(ellipse_iou, ring_iou)
    ellipse_fit_error = 1.0 - fit_iou

    stripe = pre["stripe_removed"].astype(np.float32)
    raw_l = pre["raw_l"].astype(np.float32)
    response = pre["dark_response"].astype(np.float32)
    grad = pre["gradient"].astype(np.float32)
    grad_u8 = pre["gradient_map"]
    gx = pre["gx"]
    gy = pre["gy"]
    structure = pre["structure_mask"]

    inner_vals = stripe[ellipse_inner > 0]
    outer_vals = stripe[outer > 0]
    ring_vals = grad[ring > 0]
    outer_grad_vals = grad[outer > 0]
    response_vals = response[ellipse_full > 0]
    local_vals = stripe[ellipse_outer > 0]
    raw_vals = raw_l[ellipse_full > 0]

    if inner_vals.size < 20 or outer_vals.size < 20 or ring_vals.size < 20:
        return None

    _, local_sigma = robust_stats(local_vals)
    local_sigma = max(local_sigma, 3.0)
    inner_mean = float(np.mean(inner_vals))
    outer_mean = float(np.mean(outer_vals))
    contrast = float((outer_mean - inner_mean) / local_sigma)  # positive for dark oval

    outer_grad_ref = float(np.median(outer_grad_vals)) if outer_grad_vals.size else 0.0
    grad_strength = float((np.mean(ring_vals) - outer_grad_ref) / max(float(np.std(outer_grad_vals) + 1e-6), 4.0))
    continuity, ring_strength_raw = ellipse_ring_continuity(grad, ellipse, outer_grad_ref)
    ring_strength = float(ring_strength_raw / max(float(np.std(outer_grad_vals) + 1e-6), 4.0))

    x0 = cx - major / 2.0
    x1 = cx + major / 2.0
    y0 = cy - minor / 2.0
    y1 = cy + minor / 2.0
    margin = cfg.border_margin_frac * min(h, w)
    border_touch = float(x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin)
    min_dist = min(x0, y0, w - x1, h - y1)
    border_proximity = clip01((0.090 * min(h, w) - min_dist) / max(0.090 * min(h, w), 1.0))

    structure_overlap = float(np.sum((ellipse_full > 0) & (structure > 0)) / max(np.sum(ellipse_full > 0), 1))
    anisotropy = gradient_orientation_anisotropy(gx, gy, ring)
    raw_dynamic_range = float(np.percentile(raw_vals, 95) - np.percentile(raw_vals, 5)) if raw_vals.size else 0.0
    raw_extreme_fraction = float(np.mean((raw_vals < 32) | (raw_vals > 230))) if raw_vals.size else 0.0
    edge_density = float(np.mean(grad_u8[ellipse_full > 0] > 178)) if np.any(ellipse_full > 0) else 0.0

    solidity = 0.0
    contour_area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area > 1e-6:
        solidity = contour_area / hull_area

    # Scores are kept interpretable and then combined. The final threshold is
    # learned from bad/good folders, so these are not absolute production limits.
    axis_score = clip01((cfg.max_axis_ratio - axis_ratio) / (cfg.max_axis_ratio - 1.0))
    ellipse_score = clip01((fit_iou - 0.15) / 0.38)
    solidity_score = clip01((solidity - 0.30) / 0.45)
    shape_score = 0.45 * axis_score + 0.40 * ellipse_score + 0.15 * solidity_score

    contrast_score = clip01((contrast - 0.10) / 1.65)
    grad_score = clip01((grad_strength + 0.05) / 1.25)
    ring_score = clip01((ring_strength + 0.03) / 1.10)
    continuity_score = clip01((continuity - 0.18) / 0.48)
    blob_score = clip01((float(np.mean(response_vals)) - 0.30) / 2.2)

    line_penalty = clip01((anisotropy - 0.64) / 0.28) * 12.0
    structure_penalty = clip01((structure_overlap - 0.10) / 0.30) * 20.0
    border_penalty = border_touch * 9.0 + border_proximity * 28.0
    huge_line_penalty = 8.0 if (axis_ratio > 4.5 and continuity < 0.35) else 0.0
    raw_range_penalty = clip01((raw_dynamic_range - 105.0) / 75.0) * 14.0
    raw_extreme_penalty = clip01((raw_extreme_fraction - 0.035) / 0.18) * 22.0
    edge_density_penalty = clip01((edge_density - 0.16) / 0.30) * 12.0

    final_score = (
        100.0
        * (
            0.17 * shape_score
            + 0.28 * contrast_score
            + 0.15 * ring_score
            + 0.11 * grad_score
            + 0.13 * continuity_score
            + 0.16 * blob_score
        )
        - line_penalty
        - structure_penalty
        - border_penalty
        - huge_line_penalty
        - raw_range_penalty
        - raw_extreme_penalty
        - edge_density_penalty
    )
    final_score = max(cfg.weak_score_floor, float(final_score))

    return {
        "final_score": final_score,
        "area": area,
        "major_axis": float(major),
        "minor_axis": float(minor),
        "axis_ratio": float(axis_ratio),
        "eccentricity": float(math.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-6)))),
        "solidity": float(solidity),
        "contrast": float(contrast),
        "ring_strength": float(ring_strength),
        "gradient_strength": float(grad_strength),
        "ellipse_fit_error": float(ellipse_fit_error),
        "center_x": float(cx),
        "center_y": float(cy),
        "ring_continuity": float(continuity),
        "structure_overlap": float(structure_overlap),
        "orientation_anisotropy": float(anisotropy),
        "border_touching": float(border_touch),
        "border_proximity": float(border_proximity),
        "raw_dynamic_range": float(raw_dynamic_range),
        "raw_extreme_fraction": float(raw_extreme_fraction),
        "edge_density": float(edge_density),
        "ellipse_iou": float(ellipse_iou),
        "ring_iou": float(ring_iou),
        "ellipse_angle": float(angle),
        "shape_score": float(shape_score),
        "contrast_score": float(contrast_score),
        "ring_score": float(ring_score),
        "gradient_score": float(grad_score),
        "continuity_score": float(continuity_score),
        "blob_score": float(blob_score),
    }


def detect_candidates(pre: Dict[str, np.ndarray], cfg: Config) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    candidate_mask, comp_masks = candidate_masks_from_response(pre, cfg)
    scored: List[Dict[str, float]] = []
    seen = []
    for cm in comp_masks:
        metrics = score_candidate(cm, pre, cfg)
        if metrics is None:
            continue
        key = (round(metrics["center_x"] / 6), round(metrics["center_y"] / 6), round(metrics["major_axis"] / 8), round(metrics["minor_axis"] / 8))
        if key in seen:
            continue
        seen.append(key)
        scored.append(metrics)
    scored.sort(key=lambda d: d["final_score"], reverse=True)
    return candidate_mask, scored[: cfg.max_candidates_per_image]


def draw_overlay(
    bgr: np.ndarray,
    candidates: List[Dict[str, float]],
    threshold: Optional[float],
    label: str,
    reason: str,
) -> Tuple[np.ndarray, np.ndarray]:
    overlay = bgr.copy()
    ellipse_overlay = bgr.copy()
    for i, c in enumerate(candidates[:8]):
        color = (0, 0, 255) if i == 0 else (0, 165, 255)
        if threshold is not None and c["final_score"] < threshold:
            color = (0, 220, 220)
        center = (int(round(c["center_x"])), int(round(c["center_y"])))
        axes = (int(round(c["major_axis"] / 2.0)), int(round(c["minor_axis"] / 2.0)))
        cv2.ellipse(ellipse_overlay, center, axes, float(c["ellipse_angle"]), 0, 360, color, 2)
        cv2.ellipse(overlay, center, axes, float(c["ellipse_angle"]), 0, 360, color, 2)
        cv2.putText(
            overlay,
            f"{c['final_score']:.1f}",
            (max(3, center[0] - 22), max(16, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    status_color = (0, 0, 255) if label == "oval_defect" else ((0, 200, 255) if label == "borderline" else (0, 180, 0))
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, 54), (0, 0, 0), -1)
    cv2.putText(overlay, f"{label}  {reason[:90]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
    if threshold is not None:
        cv2.putText(overlay, f"threshold={threshold:.2f}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return ellipse_overlay, overlay


def empty_metrics(filename: str, folder: str, width: int, height: int, error: str = "") -> Dict[str, object]:
    return {
        "filename": filename,
        "folder": folder,
        "width": width,
        "height": height,
        "final_label": "error" if error else "good",
        "final_score": 0.0,
        "oval_candidate_count": 0,
        "best_candidate_area": 0.0,
        "best_candidate_major_axis": 0.0,
        "best_candidate_minor_axis": 0.0,
        "best_candidate_axis_ratio": 0.0,
        "best_candidate_eccentricity": 0.0,
        "best_candidate_solidity": 0.0,
        "best_candidate_contrast": 0.0,
        "best_candidate_ring_strength": 0.0,
        "best_candidate_gradient_strength": 0.0,
        "best_candidate_ellipse_fit_error": 1.0,
        "best_candidate_center_x": 0.0,
        "best_candidate_center_y": 0.0,
        "threshold_used": np.nan,
        "decision_reason": error or "no oval-like candidate",
        "ring_continuity": 0.0,
        "structure_overlap": 0.0,
        "orientation_anisotropy": 0.0,
        "border_touching": 0.0,
        "overlay_path": "",
    }


def classify_from_score(score: float, threshold: Optional[float], margin: float) -> Tuple[str, str]:
    if threshold is None or not np.isfinite(threshold):
        return "unvalidated", "threshold unavailable: need both bad and good validation samples"
    if abs(score - threshold) <= margin:
        return "borderline", f"score {score:.2f} is within borderline margin {margin:.2f}"
    if score >= threshold:
        return "oval_defect", f"score {score:.2f} >= threshold {threshold:.2f}"
    return "good", f"score {score:.2f} < threshold {threshold:.2f}"


def process_one(
    image_path: Path,
    folder: str,
    out_root: Path,
    cfg: Config,
    threshold: Optional[float] = None,
    margin: float = 0.0,
    save_images: bool = True,
) -> Dict[str, object]:
    bgr0 = imread_unicode(image_path)
    if bgr0 is None:
        return empty_metrics(image_path.name, folder, 0, 0, "cannot read image")
    orig_h, orig_w = bgr0.shape[:2]
    bgr, scale_x, scale_y = resize_for_processing(bgr0, cfg.process_size)

    try:
        pre = preprocess(bgr, cfg)
        candidate_mask, candidates = detect_candidates(pre, cfg)
        best = candidates[0] if candidates else None
        score = float(best["final_score"]) if best else 0.0
        label, reason = classify_from_score(score, threshold, margin)
        if threshold is None:
            label = "unvalidated"
        ellipse_overlay, final_overlay = draw_overlay(bgr, candidates, threshold, label, reason)

        base_dir = out_root / "processed_images" / folder / image_path.stem
        overlay_path = base_dir / "final_overlay.png"
        if save_images:
            ensure_dir(base_dir)
            imwrite_unicode(base_dir / "original_resized.png", bgr)
            imwrite_unicode(base_dir / "cropped_roi.png", bgr)
            imwrite_unicode(base_dir / "grayscale.png", pre["gray"])
            imwrite_unicode(base_dir / "illumination_corrected.png", pre["illumination_corrected"])
            imwrite_unicode(base_dir / "contrast_enhanced.png", pre["contrast_enhanced"])
            imwrite_unicode(base_dir / "stripe_removed.png", pre["stripe_removed"])
            imwrite_unicode(base_dir / "dark_response.png", pre["dark_response_u8"])
            imwrite_unicode(base_dir / "gradient_map.png", pre["gradient_map"])
            imwrite_unicode(base_dir / "structure_mask.png", pre["structure_mask"])
            imwrite_unicode(base_dir / "candidate_mask.png", candidate_mask)
            imwrite_unicode(base_dir / "ellipse_overlay.png", ellipse_overlay)
            imwrite_unicode(overlay_path, final_overlay)

        row = empty_metrics(image_path.name, folder, orig_w, orig_h)
        row.update(
            {
                "final_label": label,
                "final_score": score,
                "oval_candidate_count": len(candidates),
                "threshold_used": threshold if threshold is not None else np.nan,
                "decision_reason": reason,
                "overlay_path": str(overlay_path),
            }
        )
        if best:
            # Map geometry back to original image coordinates.
            row.update(
                {
                    "best_candidate_area": best["area"] * scale_x * scale_y,
                    "best_candidate_major_axis": best["major_axis"] * (scale_x + scale_y) / 2.0,
                    "best_candidate_minor_axis": best["minor_axis"] * (scale_x + scale_y) / 2.0,
                    "best_candidate_axis_ratio": best["axis_ratio"],
                    "best_candidate_eccentricity": best["eccentricity"],
                    "best_candidate_solidity": best["solidity"],
                    "best_candidate_contrast": best["contrast"],
                    "best_candidate_ring_strength": best["ring_strength"],
                    "best_candidate_gradient_strength": best["gradient_strength"],
                    "best_candidate_ellipse_fit_error": best["ellipse_fit_error"],
                    "best_candidate_center_x": best["center_x"] * scale_x,
                    "best_candidate_center_y": best["center_y"] * scale_y,
                    "ring_continuity": best["ring_continuity"],
                    "structure_overlap": best["structure_overlap"],
                    "orientation_anisotropy": best["orientation_anisotropy"],
                    "border_touching": best["border_touching"],
                    "shape_score": best["shape_score"],
                    "contrast_score": best["contrast_score"],
                    "ring_score": best["ring_score"],
                    "gradient_score": best["gradient_score"],
                    "continuity_score": best["continuity_score"],
                    "blob_score": best["blob_score"],
                }
            )
        return row
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        return empty_metrics(image_path.name, folder, orig_w, orig_h, f"{exc}; {tb}")


def threshold_sweep(df: pd.DataFrame) -> Tuple[Optional[float], pd.DataFrame, Dict[str, object], float]:
    labeled = df[df["folder"].isin(["bad", "good"])].copy()
    bad_count = int(np.sum(labeled["folder"] == "bad"))
    good_count = int(np.sum(labeled["folder"] == "good"))
    if bad_count < CONFIG.min_validation_bad or good_count < CONFIG.min_validation_good:
        return None, pd.DataFrame(), {
            "validation_status": "INSUFFICIENT_SAMPLES",
            "message": f"Need at least {CONFIG.min_validation_bad} bad and {CONFIG.min_validation_good} good images; got bad={bad_count}, good={good_count}.",
        }, CONFIG.borderline_margin_abs

    scores = labeled["final_score"].astype(float).to_numpy()
    labels_bad = (labeled["folder"].to_numpy() == "bad")
    unique = sorted(set(float(s) for s in scores))
    if not unique:
        return None, pd.DataFrame(), {"validation_status": "FAILED", "message": "No scores available."}, CONFIG.borderline_margin_abs
    thresholds = []
    thresholds.append(min(unique) - 1e-6)
    for a, b in zip(unique[:-1], unique[1:]):
        thresholds.append((a + b) / 2.0)
    thresholds.append(max(unique) + 1e-6)

    rows = []
    for thr in thresholds:
        pred_bad = scores >= thr
        tp = int(np.sum(pred_bad & labels_bad))
        fn = int(np.sum((~pred_bad) & labels_bad))
        fp = int(np.sum(pred_bad & (~labels_bad)))
        tn = int(np.sum((~pred_bad) & (~labels_bad)))
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        accuracy = (tp + tn) / max(len(scores), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        balanced = (recall + specificity) / 2.0
        # Industrial preference: when recall is acceptable, reduce false alarms
        # on hard good examples; otherwise fall back to balanced separation.
        if recall >= CONFIG.min_bad_recall_preference:
            objective = balanced + 0.08 * specificity + 0.03 * precision - 0.002 * fp
        else:
            objective = balanced - 0.03 * fp
        rows.append(
            {
                "threshold": thr,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "bad_recall": recall,
                "good_specificity": specificity,
                "precision": precision,
                "accuracy": accuracy,
                "f1": f1,
                "balanced_accuracy": balanced,
                "objective": objective,
            }
        )
    sweep = pd.DataFrame(rows).sort_values(["objective", "good_specificity", "bad_recall"], ascending=[False, False, False])
    best = sweep.iloc[0].to_dict()
    threshold = float(best["threshold"])
    score_range = float(np.percentile(scores, 95) - np.percentile(scores, 5))
    margin = max(CONFIG.borderline_margin_abs, CONFIG.borderline_margin_rel * max(score_range, 1.0))

    status = "PASS"
    messages = []
    if best["bad_recall"] < 0.80 or best["good_specificity"] < 0.80 or best["balanced_accuracy"] < 0.75:
        status = "FAILED"
        messages.append("Bad/good score distributions are not reliably separable with the current CV metrics.")
    elif best["bad_recall"] < 0.90:
        status = "RISK"
        messages.append("Bad recall is below 90%; weak oval defects may be missed.")
    if status != "FAILED" and best["good_specificity"] < 0.90:
        status = "RISK"
        messages.append("Good specificity is below 90%; hard negative good images may false-alarm.")
    if bad_count < 20 or good_count < 20:
        messages.append("Sample count is small; threshold is sample-specific.")

    summary = {
        "validation_status": status,
        "message": " ".join(messages) if messages else "Bad/good score distributions are separable on the provided sample set.",
        "threshold": threshold,
        "borderline_margin": margin,
        "bad_count": bad_count,
        "good_count": good_count,
        "tp": int(best["tp"]),
        "fn": int(best["fn"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "bad_recall": float(best["bad_recall"]),
        "good_specificity": float(best["good_specificity"]),
        "precision": float(best["precision"]),
        "accuracy": float(best["accuracy"]),
        "f1": float(best["f1"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
    }
    return threshold, sweep.sort_values("threshold"), summary, margin


def apply_threshold_to_rows(rows: List[Dict[str, object]], threshold: Optional[float], margin: float) -> List[Dict[str, object]]:
    updated = []
    for row in rows:
        if row.get("final_label") == "error":
            updated.append(row)
            continue
        label, reason = classify_from_score(float(row.get("final_score", 0.0)), threshold, margin)
        row = dict(row)
        row["final_label"] = label
        row["threshold_used"] = threshold if threshold is not None else np.nan
        row["decision_reason"] = reason
        updated.append(row)
    return updated


def save_validation_outputs(df: pd.DataFrame, sweep: pd.DataFrame, summary: Dict[str, object], out_root: Path, threshold: Optional[float], margin: float) -> None:
    val_dir = out_root / "validation"
    ensure_dir(val_dir)
    ensure_dir(val_dir / "false_positive_examples")
    ensure_dir(val_dir / "false_negative_examples")
    ensure_dir(val_dir / "borderline_examples")

    if not sweep.empty:
        sweep.to_csv(val_dir / "threshold_sweep.csv", index=False)

    labeled = df[df["folder"].isin(["bad", "good"])].copy()
    if threshold is not None and not labeled.empty:
        pred_bad = labeled["final_score"].astype(float) >= threshold
        true_bad = labeled["folder"] == "bad"
        tp = int(np.sum(pred_bad & true_bad))
        fn = int(np.sum((~pred_bad) & true_bad))
        fp = int(np.sum(pred_bad & (~true_bad)))
        tn = int(np.sum((~pred_bad) & (~true_bad)))
        cm = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["actual_bad", "actual_good"],
            columns=["pred_bad", "pred_good"],
        )
    else:
        cm = pd.DataFrame()
    cm.to_csv(val_dir / "confusion_matrix.csv")

    fp_list = []
    fn_list = []
    borderline = []
    if threshold is not None:
        for _, r in labeled.iterrows():
            score = float(r["final_score"])
            is_bad = r["folder"] == "bad"
            pred = score >= threshold
            if pred and not is_bad:
                fp_list.append(str(r["filename"]))
                copy_example(r, val_dir / "false_positive_examples")
            if (not pred) and is_bad:
                fn_list.append(str(r["filename"]))
                copy_example(r, val_dir / "false_negative_examples")
            if abs(score - threshold) <= margin:
                borderline.append(str(r["filename"]))
                copy_example(r, val_dir / "borderline_examples")

    summary_lines = [
        "Oval defect validation summary",
        "================================",
    ]
    for k, v in summary.items():
        summary_lines.append(f"{k}: {v}")
    summary_lines += [
        "",
        "False positives:",
        ", ".join(fp_list) if fp_list else "none",
        "",
        "False negatives:",
        ", ".join(fn_list) if fn_list else "none",
        "",
        "Borderline validation examples:",
        ", ".join(borderline) if borderline else "none",
        "",
        "Recommended extra samples:",
        "- weak oval defect",
        "- oval defect in dark region",
        "- oval defect in gold region",
        "- oval defect crossing gold/dark boundary",
        "- good hard negatives with smooth color nonuniformity but no oval ring",
        "- good hard negatives with dense pad/trace/black-line structures",
    ]
    (val_dir / "validation_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    if plt is not None and not sweep.empty:
        try:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)
            ax.plot(sweep["threshold"], sweep["bad_recall"], label="bad recall")
            ax.plot(sweep["threshold"], sweep["good_specificity"], label="good specificity")
            ax.plot(sweep["threshold"], sweep["precision"], label="precision")
            if threshold is not None:
                ax.axvline(threshold, color="red", linestyle="--", label=f"selected {threshold:.2f}")
            ax.set_xlabel("final_score threshold")
            ax.set_ylabel("metric")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(val_dir / "threshold_sweep.png", dpi=140)
            plt.close(fig)
        except Exception:
            pass


def copy_example(row: pd.Series, target_dir: Path) -> None:
    overlay = Path(str(row.get("overlay_path", "")))
    if overlay.exists():
        ensure_dir(target_dir)
        shutil.copy2(overlay, target_dir / f"{row['folder']}_{Path(str(row['filename'])).stem}_overlay.png")


def write_metrics(rows: List[Dict[str, object]], out_root: Path) -> pd.DataFrame:
    metrics_dir = out_root / "metrics"
    ensure_dir(metrics_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame()
    for folder in ["bad", "good", "unknown"]:
        part = df[df["folder"] == folder] if not df.empty and "folder" in df.columns else pd.DataFrame()
        part.to_csv(metrics_dir / f"metrics_{folder}.csv", index=False)
    df.to_csv(metrics_dir / "metrics_all.csv", index=False)
    return df


def rerender_overlays(
    root: Path,
    out_root: Path,
    rows: List[Dict[str, object]],
    cfg: Config,
    threshold: Optional[float],
    margin: float,
    unknown_validated: bool,
) -> List[Dict[str, object]]:
    # Reprocess once after threshold selection so overlays and labels carry the
    # learned threshold instead of the temporary unvalidated label.
    final_rows = []
    for row in tqdm(rows, desc="Rendering final overlays"):
        folder = str(row["folder"])
        img_path = root / folder / str(row["filename"])
        folder_threshold = threshold
        folder_margin = margin
        if folder == "unknown" and not unknown_validated:
            folder_threshold = None
            folder_margin = 0.0
        final_rows.append(process_one(img_path, folder, out_root, cfg, folder_threshold, folder_margin, save_images=True))
    return final_rows


def run(root: Path, cfg: Config) -> int:
    out_root = root / "output"
    ensure_dir(out_root)
    ensure_dir(out_root / "processed_images")
    ensure_dir(out_root / "metrics")
    ensure_dir(out_root / "validation")

    files_by_folder = {folder: list_images(root / folder) for folder in ["bad", "good", "unknown"]}
    total = sum(len(v) for v in files_by_folder.values())
    if total == 0:
        print(f"No images found under {root}. Expected bad/good/unknown subfolders.")
        return 2

    print("Input image counts:", {k: len(v) for k, v in files_by_folder.items()})

    # First pass extracts scores without using labels to set a threshold.
    rows: List[Dict[str, object]] = []
    for folder in ["bad", "good", "unknown"]:
        for path in tqdm(files_by_folder[folder], desc=f"Scoring {folder}"):
            rows.append(process_one(path, folder, out_root, cfg, threshold=None, margin=0.0, save_images=True))

    df_first = write_metrics(rows, out_root)
    threshold, sweep, summary, margin = threshold_sweep(df_first)

    # Second pass writes final overlays/labels using selected threshold.
    unknown_validated = summary.get("validation_status") == "PASS"
    final_rows = rerender_overlays(root, out_root, rows, cfg, threshold, margin, unknown_validated)
    df = write_metrics(final_rows, out_root)
    save_validation_outputs(df, sweep, summary, out_root, threshold, margin)

    print("")
    print("Oval defect detection summary")
    print("=============================")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if threshold is None:
        print("Unknown images were processed but labeled unvalidated because bad/good validation was insufficient.")
    elif not unknown_validated:
        print("Unknown images were processed but labeled unvalidated because bad/good validation did not pass.")
    else:
        unknown = df[df["folder"] == "unknown"]
        if not unknown.empty:
            print("Unknown predictions:")
            print(unknown["final_label"].value_counts(dropna=False).to_string())
    print(f"Output folder: {out_root}")
    return 0 if summary.get("validation_status") in {"PASS", "RISK"} else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect oval shadow/ring defects in Fab AOI images.")
    parser.add_argument("--root", default=r"C:\defect_ai", help=r"Root folder containing bad, good, unknown. Default: C:\defect_ai")
    parser.add_argument("--process-size", type=int, default=CONFIG.process_size, help="Processing image size. Use 482 for production.")
    parser.add_argument("--min-area-frac", type=float, default=CONFIG.min_area_frac)
    parser.add_argument("--max-area-frac", type=float, default=CONFIG.max_area_frac)
    parser.add_argument("--max-axis-ratio", type=float, default=CONFIG.max_axis_ratio)
    parser.add_argument("--no-row-normalize", action="store_true", help="Disable horizontal stripe suppression.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = Config(**asdict(CONFIG))
    cfg.process_size = args.process_size
    cfg.min_area_frac = args.min_area_frac
    cfg.max_area_frac = args.max_area_frac
    cfg.max_axis_ratio = args.max_axis_ratio
    cfg.row_normalize = not args.no_row_normalize
    return run(Path(args.root), cfg)


if __name__ == "__main__":
    raise SystemExit(main())
