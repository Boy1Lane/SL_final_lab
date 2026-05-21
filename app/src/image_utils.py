from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def read_uploaded_image(uploaded_file) -> Image.Image:
    return read_image_bytes(uploaded_file)


def read_image_bytes(image_source) -> Image.Image:
    try:
        image = Image.open(image_source)
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG. Vui lòng chọn ảnh khác.") from exc


def read_data_url_image(data_url: str) -> Image.Image:
    if not data_url.startswith("data:image/"):
        raise ValueError("Clipboard không chứa ảnh hợp lệ. Hãy copy screenshot hoặc ảnh JPG/PNG rồi thử lại.")

    try:
        _, encoded = data_url.split(",", 1)
        return read_image_bytes(BytesIO(base64.b64decode(encoded)))
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Không thể đọc ảnh từ clipboard. Hãy copy lại ảnh rồi thử Ctrl+V lần nữa.") from exc


def save_result_image(image, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"result_{timestamp}.png"
    suffix = 2
    while file_path.exists():
        file_path = output_path / f"result_{timestamp}_{suffix}.png"
        suffix += 1
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
        raise ValueError("Ảnh phải có 3 kênh màu RGB.")
    return array
