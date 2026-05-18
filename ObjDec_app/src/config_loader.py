from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when model configuration is invalid."""


def load_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"File JSON không hợp lệ: {json_path}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"File JSON phải có dạng object: {json_path}")
    return data


def load_inference_config(model_dir: str | Path) -> dict:
    model_path = Path(model_dir)
    class_names = load_json(model_path / "class_names.json")
    config = load_json(model_path / "inference_config.json")

    _validate_class_names(class_names, model_path / "class_names.json")
    _validate_inference_config(config, model_path / "inference_config.json")

    normalized_class_names = {int(class_id): name for class_id, name in class_names.items()}
    config["class_names"] = normalized_class_names
    return config


def _validate_class_names(class_names: dict[str, Any], path: Path) -> None:
    if not class_names:
        raise ConfigError(f"class_names.json không được rỗng: {path}")

    for class_id, class_name in class_names.items():
        try:
            int(class_id)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Class id phải là số trong {path}: {class_id}") from exc
        if not isinstance(class_name, str) or not class_name.strip():
            raise ConfigError(f"Class name không hợp lệ trong {path}: {class_id}")


def _validate_inference_config(config: dict[str, Any], path: Path) -> None:
    required_fields = {
        "model_name": str,
        "model_file": str,
        "image_size": int,
        "confidence_threshold": (float, int),
        "iou_threshold": (float, int),
        "classes": list,
    }

    for field, expected_type in required_fields.items():
        if field not in config:
            raise ConfigError(f"Thiếu field `{field}` trong {path}")
        if not isinstance(config[field], expected_type):
            raise ConfigError(f"Field `{field}` sai kiểu dữ liệu trong {path}")

    if not 0 < float(config["confidence_threshold"]) < 1:
        raise ConfigError("confidence_threshold phải nằm trong khoảng 0..1")
    if not 0 < float(config["iou_threshold"]) < 1:
        raise ConfigError("iou_threshold phải nằm trong khoảng 0..1")
    if int(config["image_size"]) <= 0:
        raise ConfigError("image_size phải lớn hơn 0")
    if not all(isinstance(item, str) and item.strip() for item in config["classes"]):
        raise ConfigError("classes phải là danh sách string không rỗng")

    device = config.get("device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device chỉ được là auto, cpu hoặc cuda")
    config["device"] = device
