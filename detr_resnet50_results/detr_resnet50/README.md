# DETR ResNet-50 Object Detection Result

## Model

- Model: `facebook/detr-resnet-50`
- Framework: PyTorch + Hugging Face Transformers
- Task: Traffic object detection
- Number of classes: `5`
- Classes: `['person', 'car', 'bus', 'bicycle', 'motorbike']`

## Dataset

- Dataset path: `/content/subset_3000_500_1000`
- COCO path: `/content/subset_3000_500_1000/coco`
- Train annotation: `/content/subset_3000_500_1000/coco/annotations/instances_train.json`
- Val annotation: `/content/subset_3000_500_1000/coco/annotations/instances_val.json`
- Test annotation: `/content/subset_3000_500_1000/coco/annotations/instances_test.json`

## Training Config

- Batch size: `4`
- Epochs: `10`
- Learning rate: `1e-05`
- Weight decay: `0.0001`
- Eval confidence threshold: `0.001`
- Prediction confidence threshold: `0.25`
- Device: `cuda`
- Seed: `42`

## Evaluation Metrics

Main metrics saved in `metrics/evaluation_metrics.csv` and `metrics/evaluation_metrics.json`.

Required metrics:

- mAP@0.5
- mAP@0.5:0.95
- Inference time
- FPS
- Number of parameters
- Model size

## Output Structure

```text
model_results/detr_resnet50/
├── config/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── training_config.json
├── checkpoints/
│   ├── best_detr_model/
│   └── final_detr_model/
├── metrics/
│   ├── evaluation_metrics.csv
│   ├── evaluation_metrics.json
│   ├── per_class_metrics.csv
│   └── training_history.csv
├── predictions/
│   ├── sample_predictions/
│   └── prediction_examples.csv
├── logs/
│   └── training_log.txt
└── README.md
```

## Notes

- This notebook maps original COCO `category_id` values to contiguous DETR labels `0..N-1`.
- mAP evaluation uses a low confidence threshold to avoid discarding low-confidence detections before metric computation.
- Prediction examples use a separate visualization threshold.