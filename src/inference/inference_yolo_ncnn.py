import sys
import logging as log
from pathlib import Path

import cv2
import numpy as np
#from ultralytics.yolo.utils.ops import scale_image

from .models.yolo_ncnn import YoloNCNN

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('configs')))
from models_configs.model_configurator import ModelConfig
from logger_conf import configure_logger


configure_logger()
ROOT_DIR = Path(__file__).resolve().parents[2]

log.info('[NCNN] Setting up models config')
MODEL_CONF = ModelConfig()

DETECT_MODEL_WEIGHTS = ROOT_DIR / MODEL_CONF.properties['detection']['yolov8n']['ncnn']['model_dir']
log.info(f'[NCNN] Loading detection model: {DETECT_MODEL_WEIGHTS}')
NCNN_MODEL_DET = YoloNCNN(DETECT_MODEL_WEIGHTS, 'detect')

SEGMENT_MODEL_WEIGHTS = ROOT_DIR / MODEL_CONF.properties['segmentation']['yolov8n']['ncnn']['model_dir']
log.info(f'[NCNN] Loading segmentation model: {SEGMENT_MODEL_WEIGHTS}')
NCNN_MODEL_SEG = YoloNCNN(SEGMENT_MODEL_WEIGHTS, 'segment')


def detect_get_bounding_boxes(input, raw_output=False):
    """
    Performs NCNN inference on input. Can return raw boxes coodrinates
    (use raw_output flag) or rectangles as bounding boxes.
    Params:
        input: image. np.ndarray
    Returns:
        input_image: The combined image. np.ndarray
    """

    log.info('Inference: detection')
    results = NCNN_MODEL_DET.simple_predict(input)
    out_boxes = []
    rects = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        out_boxes += boxes
        for box in boxes:
            rects.append(box.xyxy[0].astype(int))

    log.info(F'Detected {len(rects)} objects')
    if raw_output:
        return out_boxes

    return rects

def detect_draw_boxes(input_image):
    """
    Combines image and its detected bounding boxes into a single image.
    Params:
        input_image: image. np.ndarray
    Returns:
        input_image: The combined image. np.ndarray
    """
    log.info('Drawing bounding boxes')
    rects = detect_get_bounding_boxes(input_image)
    for r in rects:
        cv2.rectangle(input_image, r[:2], r[2:], (255, 255, 255), 2)
    return input_image


def segment_predict_on_image(img, conf):
    result = NCNN_MODEL_SEG.simple_predict(img, conf=conf)[0]

    if result.masks is None:
        return None

    # segmentation
    masks = result.masks.data.cpu().numpy()

    return masks


def segment_draw_overlay(image, color=(0,255,0), alpha=0.3, resize=None):
    """
    Combines image and its segmentation mask into a single image.

    Params:
        image: image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """

    masks = segment_predict_on_image(image, conf=0.55)
    h, w, _ = image.shape

    for mask in masks:
        mask = cv2.resize(mask, (w, h))
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image
