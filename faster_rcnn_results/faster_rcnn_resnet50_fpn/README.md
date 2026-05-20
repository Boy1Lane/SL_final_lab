# Faster R-CNN ResNet50-FPN Result

## Dataset
- Dataset: VOC0712 traffic subset 3000/500/1000
- Classes: person, car, bus, bicycle, motorbike
- Train/Val/Test: 3000/500/1000

## Model
- Architecture: Faster R-CNN ResNet50-FPN (torchvision)
- Pretrained: Yes
- Epochs: 10
- Batch size: 2
- Optimizer: SGD (LR=0.005)
- Freeze backbone: No
- Device: CUDA

## Output
- Best checkpoint: checkpoints/best_model.pth
- Final checkpoint: checkpoints/final_model.pth
- Main metrics: metrics/evaluation_metrics.csv
- Per-class metrics: metrics/per_class_metrics.csv
- Prediction examples: predictions/sample_predictions/
