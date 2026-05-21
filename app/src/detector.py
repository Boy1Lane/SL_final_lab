from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.config_loader import load_inference_config


class ModelNotFoundError(FileNotFoundError):
    """Raised when the YOLO model file is missing."""


class YOLODetector:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.config = load_inference_config(self.model_dir)
        self.model_path = self.model_dir / self.config["model_file"]
        self.display_model_path = Path("app_models") / self.model_dir.name / self.config["model_file"]
        self.class_names: dict[int, str] = self.config["class_names"]

        if not self.model_path.exists():
            raise ModelNotFoundError(f"Không tìm thấy model. Hãy đặt file best.pt vào {self.display_model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Thiếu thư viện ultralytics. Hãy chạy: pip install -r requirements.txt") from exc

        self.model = YOLO(str(self.model_path))

    def predict(self, image: Any, conf: float, iou: float, imgsz: int, device: str) -> dict:
        selected_device = self._resolve_device(device)
        started_at = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=selected_device,
            verbose=False,
        )
        inference_time_ms = (time.perf_counter() - started_at) * 1000

        if not results:
            return {
                "annotated_image": self._to_rgb_array(image),
                "detections": [],
                "device": selected_device,
                "inference_time_ms": inference_time_ms,
            }

        result = results[0]
        detections = self._extract_detections(result)
        annotated_rgb = self._draw_detections(image, detections)

        return {
            "annotated_image": annotated_rgb,
            "detections": detections,
            "device": selected_device,
            "inference_time_ms": inference_time_ms,
        }

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if device in {"cpu", "cuda"}:
            return device
        return "cpu"

    def _extract_detections(self, result: Any) -> list[dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = [int(round(value)) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
        return detections

    def _draw_detections(self, image: Any, detections: list[dict]) -> np.ndarray:
        base = Image.fromarray(self._to_rgb_array(image)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(max(13, round(min(base.size) / 45)))
        width, height = base.size
        stroke = max(3, round(min(width, height) / 220))
        radius = max(8, round(min(width, height) / 80))

        for item in detections:
            xmin, ymin, xmax, ymax = self._clamp_bbox(item["bbox"], width, height)
            color = self._class_color(item["class_id"])
            draw.rounded_rectangle(
                [xmin, ymin, xmax, ymax],
                radius=radius,
                outline=color + (235,),
                width=stroke,
            )

            label = f'{item["class_name"]} {item["confidence"] * 100:.1f}%'
            text_box = draw.textbbox((0, 0), label, font=font)
            label_w = text_box[2] - text_box[0] + 18
            label_h = text_box[3] - text_box[1] + 12
            label_x = xmin
            label_y = max(0, ymin - label_h - 4)
            if label_y == 0:
                label_y = min(height - label_h, ymin + 4)

            draw.rounded_rectangle(
                [label_x, label_y, min(width, label_x + label_w), label_y + label_h],
                radius=6,
                fill=(3, 12, 28, 214),
                outline=color + (210,),
                width=1,
            )
            draw.text((label_x + 9, label_y + 5), label, fill=(241, 245, 249, 255), font=font)

        return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))

    @staticmethod
    def _to_rgb_array(image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @staticmethod
    def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
        xmin, ymin, xmax, ymax = bbox
        return [
            max(0, min(width - 1, xmin)),
            max(0, min(height - 1, ymin)),
            max(0, min(width - 1, xmax)),
            max(0, min(height - 1, ymax)),
        ]

    @staticmethod
    def _class_color(class_id: int) -> tuple[int, int, int]:
        palette = [
            (14, 165, 233),
            (34, 197, 94),
            (245, 158, 11),
            (168, 85, 247),
            (20, 184, 166),
            (239, 68, 68),
        ]
        return palette[class_id % len(palette)]

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        for font_path in (
            "C:/Windows/Fonts/Inter-Regular.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()
