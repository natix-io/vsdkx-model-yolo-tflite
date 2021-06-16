# -*- coding:utf-8 -*-
import cv2
import time
import tensorflow as tf
import numpy as np

from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference
from vsdkx.core.util.model import load_tflite


class YoloTfliteDriver(ModelDriver):
    """
    Class for people detection using YOLOv5 in tflite
    """

    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        """

        Args:
            model_config (dict): Model config dictionary with the
            following keys:
                'model_path' (str): Path to the tflite model
                'input_shape' (tuple): Shape of input(inference) image
                'person_class_id' (int): Person class ID
            model_settings (dict): Model settings config with the
            following keys:
                'target_shape' (tuple): Image target shape
                'iou_thresh' (float): Threshold for Intersection of Union
                'conf_threshold' (float): Confidence threshold
            drawing_config (dict): Debug config dictionary with the
            following keys:
                'text_thickness' (int): Text thickness
                'text_fontscale' (int): Text font scale
                'text_color' (tuple): Tuple of color in RGB for text
                'rectangle_color' (tuple): Tuple of color in RGB for rectangle
        """
        super().__init__(model_settings, model_config, drawing_config)
        self._input_shape = model_config['input_shape']
        self._filter_classes = model_config.get('filter_class_ids', [])
        self._interpreter, self._input_details, self._output_details = \
            load_tflite(model_config['model_path'])
        self._target_shape = model_settings['target_shape']
        self._conf_thresh = model_settings['conf_thresh']
        self._iou_thresh = model_settings['iou_thresh']
        self._text_thickness = drawing_config['text_thickness']
        self._text_fontscale = drawing_config['text_fontscale']
        self._text_color = drawing_config['text_color']
        self._rectangle_color = drawing_config['rectangle_color']

    def inference(self, image) -> Inference:
        """
        People detection with ssd mobilenet coco

        Args:
            image (np.array): 3D image array

        Returns:
            (Inference): the result of the ai
        """

        # Resize the original image for inference
        resized_image = self._resize_img(image, self._input_shape)

        inf_start = time.perf_counter()
        # Run the inference
        self._interpreter.set_tensor(
            self._input_details[0]['index'], resized_image)
        self._interpreter.invoke()

        x = self._interpreter.get_tensor(self._output_details[0]['index'])

        # Run the NMS to get the boxes with the highest confidence
        y = self._prediction_postprocessing(x)
        boxes, scores, classes = [], [], []
        if y[0] is not None:
            y = np.squeeze(y, axis=0)
            boxes, scores, classes = y[:, :4], y[:, 4:5], y[:, 5:6]
            boxes = self._scale_boxes(boxes, self._input_shape,
                                      self._target_shape)

        result_boxes = []
        result_scores = []
        result_classes = []

        if len(self._filter_classes) > 0:
            # Go through the prediction results
            for box, score, c_id in zip(boxes, scores, classes):
                # Iterate over the predicted bounding boxes and filter
                #   the boxes with class "person"
                if c_id in self._filter_classes:
                    result_boxes.append(box)
                    result_scores.append(score)
                    result_classes.append(c_id)
        else:
            result_scores = scores
            result_boxes = boxes
            result_classes = classes

        return Inference(result_boxes, result_classes, result_scores, {})

    def _resize_img(self, image, input_shape):
        """
        Resize input image to the expected input shape

        Args:
            image (np.array): 3D numpy array of input image
            input_shape (tuple): The shape of the input image

        Returns:
            (array): Resize image
        """

        image_resized = self._letterbox(image, new_shape=input_shape)[0]

        image_np = image_resized / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        image_np = tf.cast(image_np, dtype=tf.float32)

        return image_np

    def _letterbox(self, img,
                   new_shape=(640, 640),
                   color=(114, 114, 114)):
        """
        Resize image in letterbox fashion
        Args:
            img (np.array): 3D numpy array of input image
            new_shape (tuple): Array with the new image height and width
            color (tuple): Color array

        Returns:
            (np.array): np.array with the resized image
            (tuple): The height and width ratios
            (tuple): The width and height paddings
        """
        # Resize image to a 32-pixel-multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
                 new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color)  # add border

        return img, ratio, (dw, dh)

    def _scale_boxes(self, boxes, input_shape, target_shape):
        """
        Scales the boxes to the size of the target image

        Args:
            boxes (np.array): Array containing the bounding boxes
            input_shape (tuple): The shape of the resized image
            target_shape (tuple): The shape of the target image

        Returns:
            (np.array): np.array with the scaled bounding boxes
        """
        gain = min(input_shape[0] / target_shape[0],
                   input_shape[1] / target_shape[1])
        pad = (input_shape[1] - target_shape[1] * gain) / 2, \
              (input_shape[0] - target_shape[0] * gain) / 2
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :] /= gain

        return boxes

    def _prediction_postprocessing(self, prediction):
        """
        Processes the prediction results and passes them through NMS

        Args:
            prediction (np.array): Array with the post-processed
            inference predictions

        Returns:
             (np.array): np.array detections with shape
              nx6 (x1, y1, x2, y2, conf, cls)
        """
        # Get candidates with confidence higher than the threshold
        xc = prediction[..., 4] > self._conf_thresh  # candidates

        # Maximum width and height
        max_wh = 4096  # (pixels) minimum and maximum box width and height

        output = [None]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center box, center y, width, height) to (x1, y1, x2, y2)
            box = self._decode_box(x[:, :4])
            # .nonzero(as_tuple=False).T
            i, j = np.nonzero(x[:, 5:] > self._conf_thresh)
            i, j = np.transpose(i), np.transpose(j)
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), axis=1)
            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            nms = np.array(tf.image.non_max_suppression(boxes, scores,
                                                        100, self._iou_thresh,
                                                        self._conf_thresh))

            if len(nms) > 0:
                output[xi] = x[nms]

        return np.asarray(output)

    def _decode_box(self, box):
        """
        Decoding boxes from [box, y, w, h] to [x1, y1, x2, y2]
        where xy1=top-left, xy2=bottom-right

        Args:
            box (np.array): Array with box coordinates

        Returns:
            (np.array): np.array with new box coordinates
        """
        y = np.zeros_like(box)
        y[:, 0] = box[:, 0] - box[:, 2] / 2  # top left box
        y[:, 1] = box[:, 1] - box[:, 3] / 2  # top left y
        y[:, 2] = box[:, 0] + box[:, 2] / 2  # bottom right box
        y[:, 3] = box[:, 1] + box[:, 3] / 2  # bottom right y

        return y
