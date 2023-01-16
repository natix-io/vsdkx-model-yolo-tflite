# Yolov5 Tflite

This repository provides support for inference with tflite weights for yolov5.

**Important**: This is an experimental repository and it is not attached or included to any of our production ready repositories.

### Model Settings

```yaml
'conf_thresh': 0.5, # Float class confidence threshold
'iou_thresh': 0.4, # Float Intersection of Union threshold
```
 
#### Model Config

```yaml
'model_path': 'weight.tflite'
'input_shape':
  - 640
  - 640
'filter_class_ids':
    - 0
```
