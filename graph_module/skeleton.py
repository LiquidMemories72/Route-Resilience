"""One-pixel-wide road centerline generation."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from skimage.morphology import skeletonize

try:
    from .preprocess import ensure_binary
except ImportError:  # Supports direct run_pipeline.py execution.
    from preprocess import ensure_binary

PathLike = Union[str, Path]


def make_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary road mask and return a boolean array."""
    return skeletonize(ensure_binary(mask).astype(bool)).astype(bool)


def save_skeleton(skeleton: np.ndarray, path: PathLike) -> None:
    """Save a skeleton as a black-and-white PNG."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = (np.asarray(skeleton, dtype=bool) * 255).astype(np.uint8)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not write skeleton: {output}")
