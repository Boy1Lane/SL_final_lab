from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

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
            raise ModelNotFoundError(
                f"Không tìm thấy model best.pt. Vui lòng đặt model vào {self.display_model_path}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài ultralytics. Vui lòng chạy: pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(str(self.model_path))

    def predict(self, image: Any, conf: float, iou: float, imgsz: int, device: str) -> dict:
        selected_device = self._resolve_device(device)
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=selected_device,
            verbose=False,
        )

        if not results:
            return {"annotated_image": np.asarray(image), "detections": []}

        result = results[0]
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "annotated_image": annotated_rgb,
            "detections": self._extract_detections(result),
            "device": selected_device,
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
