### Model Settings

```yaml
'conf_thresh': 0.5, # Float class confidence threshold
'iou_thresh': 0.4, # Float Intersection of Union threshold
```
 
#### Model Config

Below we show the default system config settings: 

```yaml
'model_path': 'weight.tflite'
'input_shape':
  - 640
  - 640
'filter_class_ids':
    - 0
```
