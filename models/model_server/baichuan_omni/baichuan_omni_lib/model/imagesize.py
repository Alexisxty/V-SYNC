from __future__ import annotations

from typing import Tuple

from PIL import Image


def get(path: str) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size
