from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Traits:
    area_px: float
    perimeter_px: float
    major_axis_px: float
    minor_axis_px: float
    esd_px: float
    biovol_proxy: float  # very rough: (ESD^3)

def segment_simple(img_rgb: np.ndarray) -> np.ndarray:
    """
    Returns binary mask for foreground object.
    img_rgb: float [0,1] or uint8 [0,255]
    """
    if img_rgb.dtype != np.uint8:
        im = (img_rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        im = img_rgb

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure object is white (foreground). If background is white, invert.
    if mask.mean() > 127:
        mask = 255 - mask

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return (mask > 0).astype(np.uint8)

def compute_traits(mask: np.ndarray) -> Traits:
    """
    mask: 0/1 uint8
    """
    mask255 = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return Traits(0, 0, 0, 0, 0, 0)

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))

    major = minor = 0.0
    if len(c) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))

    # Equivalent spherical diameter (pixel-space)
    esd = float(np.sqrt(4.0 * area / (np.pi + 1e-12)))
    biovol = float(esd ** 3)

    return Traits(
        area_px=area,
        perimeter_px=perim,
        major_axis_px=major,
        minor_axis_px=minor,
        esd_px=esd,
        biovol_proxy=biovol,
    )

def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if img_rgb.dtype != np.uint8:
        base = (img_rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        base = img_rgb.copy()

    overlay = base.copy()
    overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * (1 - alpha) + np.array([255, 255, 255]) * alpha).astype(np.uint8)
    return overlay
