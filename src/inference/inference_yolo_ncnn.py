import sys
from pathlib import Path

import cv2

from models.yolo_ncnn import YoloNCNN

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('configs')))
from models_configs.model_configurator import ModelConfig
from logger_conf import configure_logger


log = configure_logger()
MODEL_CONF = ModelConfig()
NCNN_MODEL_DET = YoloNCNN(MODEL_CONF.properties['detection']['yolov8n']['ncnn']['model_dir'])


def get_bounding_boxes(input, raw_output=False):
    log.indo('Inference: detection')
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

def draw_boxes(input_image):
    log.indo('Drawing bounding boxes')
    rects = get_bounding_boxes(input_image)
    for r in rects:
        cv2.rectangle(input_image, r[:2], r[2:], (255, 255, 255), 2)
    return input_image
