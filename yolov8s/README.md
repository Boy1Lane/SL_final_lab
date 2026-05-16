# YOLOv8s Result

## Dataset
- Dataset: VOC0712 traffic subset 3000/500/1000
- Classes: person, car, bus, bicycle, motorbike
- Train/Val/Test: 3000/500/1000

## Model
- Architecture: YOLOv8s
- Model type: One-stage object detector
- Pretrained: yolov8s.pt
- Epochs: 30
- Batch size: 16
- Image size: 640
- Device: cuda
- GPU: Tesla T4

## Output
- Best checkpoint: checkpoints/best.pt
- Last checkpoint: checkpoints/last.pt
- Main metrics: metrics/evaluation_metrics.csv
- Per-class metrics: metrics/per_class_metrics.csv
- Training history: metrics/training_history.csv
- Prediction examples: predictions/sample_predictions/
- Prediction CSV: predictions/prediction_examples.csv
- Training log: logs/training_log.txt

## Main Results
- mAP@0.5 Pascal VOC: 0.740367061447967
- mAP@0.5:0.95 COCO: 0.5184398470692995
- Average inference time per image: 0.021939966678619385
- FPS: 45.57892063594178
- Total parameters: 11127519
- Trainable parameters: 0
- Model size MB: 21.45798683166504
