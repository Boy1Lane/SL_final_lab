from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError


def read_uploaded_image(uploaded_file) -> Image.Image:
    try:
        image = Image.open(uploaded_file)
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("File upload không phải ảnh hợp lệ.") from exc


def save_result_image(image, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"result_{timestamp}_{uuid4().hex[:8]}.png"
    Image.fromarray(_to_uint8_rgb(image)).save(file_path, format="PNG")
    return file_path


def encode_image_for_download(image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.fromarray(_to_uint8_rgb(image)).save(buffer, format=image_format)
    return buffer.getvalue()


def _to_uint8_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image phải có 3 kênh RGB.")
    return array
