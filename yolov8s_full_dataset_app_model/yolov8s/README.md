# YOLOv8s Full Dataset Model for Python Object Detection App

## Purpose

This model is trained on the full VOC0712 traffic 5-class dataset and is intended for deployment in a Python object detection application.

## Classes

- person
- car
- bus
- bicycle
- motorbike

## App Usage

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.predict("image.jpg", conf=0.25, iou=0.45, imgsz=640)
```

## Output Files

- `model/best.pt`: main model checkpoint for app deployment
- `model/last.pt`: last training checkpoint
- `config/data.yaml`: dataset config
- `config/training_config.json`: training settings
- `config/class_names.json`: class id to class name mapping
- `config/inference_config.json`: default inference settings
- `metrics/evaluation_metrics.csv`: main evaluation metrics
- `metrics/evaluation_metrics.json`: main evaluation metrics in JSON
- `metrics/per_class_metrics.csv`: per-class mAP
- `predictions/sample_predictions/`: visual prediction examples
- `predictions/prediction_examples.csv`: prediction CSV examples
- `logs/training_log.txt`: training log
- `logs/results.csv`: Ultralytics training history

## Main Metrics

- mAP@0.5: 0.8461324820427603
- mAP@0.5:0.95: 0.6355022694516625
- Precision: 0.8528524569053706
- Recall: 0.7509901776929775
- FPS: 45.83247804333762
- Model size MB: 21.45560646057129
