"""Counted, numbered candidate overlays for manual review; no biological acceptance."""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries


def review_overlay(image, labels, candidates, *, title, subtitle, muted_labels=()):
    """Return a derived RGB Pillow image with an explicit candidate count.

    Label colors and numbers match the label TIFF and manifest. No candidates
    are hidden by area or quality flags. Geometry below the banner is unchanged.
    ``muted_labels`` (e.g. quality-flagged candidates) get grey numbers.
    """
    if labels.shape != image.shape[:2]:
        raise ValueError("Overlay labels and overview dimensions must match")
    actual = set(int(v) for v in np.unique(labels) if v)
    if actual != {c["label"] for c in candidates}:
        raise ValueError("Overlay candidate list does not match label image")
    display = np.rint(image.astype(np.float32) / np.iinfo(image.dtype).max * 255)
    if display.ndim == 2:
        display = np.repeat(display[..., None], 3, axis=2)
    palette = np.zeros((int(labels.max()) + 1, 3), dtype=np.float32)
    for label in actual:
        palette[label] = (
            np.array(colorsys.hsv_to_rgb((label * 0.618034) % 1, 0.9, 0.85)) * 255
        )
    foreground = labels > 0
    display[foreground] = (
        0.82 * display[foreground] + 0.18 * palette[labels[foreground]]
    )
    borders = find_boundaries(labels, mode="inner")
    display[borders] = palette[labels[borders]]
    banner_height = 125
    width = max(image.shape[1], 800)
    result = Image.new("RGB", (width, image.shape[0] + banner_height), "#152330")
    result.paste(Image.fromarray(display.astype(np.uint8)), (0, banner_height))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    draw.text((16, 10), title, font=font, fill="white")
    draw.text(
        (16, 40),
        f"Detected candidates: {len(candidates)} | flagged (grey): "
        f"{len(set(muted_labels))} | all need review",
        font=font,
        fill="#ffdf83",
    )
    draw.text((16, 72), subtitle, font=small, fill="white")
    draw.text(
        (16, 98),
        "Colors/IDs = proposed instances. Boundaries and count are unvalidated.",
        font=small,
        fill="#d3dfeb",
    )
    for candidate in candidates:
        x, y = candidate["overview_centroid_xy"]
        x, y = int(round(x)), int(round(y))
        x, y = min(x, labels.shape[1] - 1), min(y, labels.shape[0] - 1)
        if labels[y, x] != candidate["label"]:
            x0, y0, x1, y1 = candidate["overview_bounds"]
            ys, xs = np.nonzero(labels[y0:y1, x0:x1] == candidate["label"])
            nearest = np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2)
            x, y = int(xs[nearest] + x0), int(ys[nearest] + y0)
        draw.text(
            (x, y + banner_height),
            str(candidate["label"]),
            font=small,
            fill="#9a9a9a" if candidate["label"] in muted_labels else "white",
            stroke_width=2,
            stroke_fill="black",
            anchor="mm",
        )
    return result
