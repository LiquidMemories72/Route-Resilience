"""Binary-mask loading and morphological cleanup."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def ensure_binary(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Return a 2-D uint8 mask containing only 0 and 1."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}")
    if array.dtype == bool or array.max(initial=0) <= 1:
        return (array > 0).astype(np.uint8)
    return (array > threshold).astype(np.uint8)


def load_binary_mask(path: PathLike, threshold: int = 127) -> np.ndarray:
    """Load a grayscale or color PNG-like image as a 0/1 mask."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return ensure_binary(image, threshold=threshold)


def cleanup_mask(
    mask: np.ndarray,
    closing_kernel_size: int = 5,
    closing_iterations: int = 1,
    min_component_area: int = 50,
    threshold: int = 127,
) -> np.ndarray:
    """Close small breaks/holes and discard small foreground components.

    Args:
        mask: Binary-like NumPy array.
        closing_kernel_size: Elliptical kernel diameter. Set to 0 or 1 to skip.
        closing_iterations: Number of closing passes.
        min_component_area: Components smaller than this many pixels are removed.
        threshold: Threshold used when ``mask`` is not already binary.
    """
    binary = ensure_binary(mask, threshold=threshold)

    if closing_kernel_size > 1 and closing_iterations > 0:
        if closing_kernel_size % 2 == 0:
            closing_kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations
        )

    if min_component_area > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cleaned = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned[labels == label] = 1
        binary = cleaned

    return binary.astype(np.uint8)


def save_binary_mask(mask: np.ndarray, path: PathLike) -> None:
    """Save a binary-like mask as a visible 8-bit image."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), ensure_binary(mask) * 255):
        raise OSError(f"Could not write mask: {output}")
