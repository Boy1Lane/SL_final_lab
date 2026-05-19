# YOLOv8s Result

## Dataset
- Dataset: VOC0712 traffic subset 3000/500/1000
- Dataset root: /content/subset_3000_500_1000
- YOLO data yaml: /content/subset_3000_500_1000/yolo/data.yaml
- Fixed training yaml: /content/model_results/yolov8s/config/data.yaml
- Classes: person, car, bus, bicycle, motorbike
- Train/Val/Test: 3000/500/1000

## Model
- Architecture: YOLOv8s
- Model type: One-stage object detector
- Pretrained: yolov8s.pt
- Epochs: 30
- Batch size: 16
- Image size: 640
- Optimizer: auto/default_ultralytics
- Learning rate: default_ultralytics
- Freeze backbone: No
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
- Sample image list: predictions/sample_test_images.txt
- Training log: logs/training_log.txt

## Main Results
- mAP@0.5 Pascal VOC: 0.8792838842824022
- mAP@0.5:0.95 COCO: 0.6491979948543214
- Precision: 0.8842213924921485
- Recall: 0.7849105957851963
- Average inference time per image: 0.02274696373939514
- FPS: 43.961911200839275
- Inference images used: 1000
- Total parameters: 11127519
- Trainable parameters: 0
- Model size MB: 21.45804786682129
