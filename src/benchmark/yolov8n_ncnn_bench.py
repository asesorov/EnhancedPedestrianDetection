import sys
import logging as log
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('inference')))
from inference_yolo_ncnn import detect_get_bounding_boxes, segment_predict_on_image
sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('configs')))
from models_configs.model_configurator import ModelConfig
from logger_conf import configure_logger

configure_logger()

NUM_ITERATIONS = 10
TEST_DATA_PATH = '/home/asesorov/itmo/ncnn-py/night_ped_0.png'

log.info('Starting benchmark: YOLOv8n detection PyTorch')
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_CONF = ModelConfig()
DETECT_MODEL_WEIGHTS = ROOT_DIR / MODEL_CONF.properties['detection']['yolov8n']['pytorch']['weights']
pytorch_model_det = YOLO(DETECT_MODEL_WEIGHTS)
SEGMENT_MODEL_WEIGHTS = ROOT_DIR / MODEL_CONF.properties['detection']['yolov8n']['pytorch']['weights']
pytorch_model_seg = YOLO(SEGMENT_MODEL_WEIGHTS)

pytorch_total_time_det = 0
for _ in range(NUM_ITERATIONS):
    start_time = time.time()
    res = pytorch_model_det(TEST_DATA_PATH, imgsz=(640,640))
    end_time = time.time()
    inference_time = end_time - start_time
    pytorch_total_time_det += inference_time

log.info('Starting benchmark: YOLOv8n segmentation PyTorch')
pytorch_total_time_seg = 0

for _ in range(NUM_ITERATIONS):
    start_time = time.time()
    res = pytorch_model_seg(TEST_DATA_PATH)
    end_time = time.time()
    inference_time = end_time - start_time
    pytorch_total_time_seg += inference_time


pytorch_fps_det = NUM_ITERATIONS / pytorch_total_time_det
pytorch_average_time_det = pytorch_total_time_det / NUM_ITERATIONS

pytorch_fps_seg = NUM_ITERATIONS / pytorch_total_time_seg
pytorch_average_time_seg = pytorch_total_time_seg / NUM_ITERATIONS

log.info('Starting benchmark: YOLOv8n detection NCNN')
ncnn_total_time_det = 0

for _ in range(NUM_ITERATIONS):
    start_time = time.time()
    res = detect_get_bounding_boxes(TEST_DATA_PATH, imgsz=(640,640))
    end_time = time.time()
    inference_time = end_time - start_time
    ncnn_total_time_det += inference_time

log.info('Starting benchmark: YOLOv8n segmentation NCNN')
ncnn_total_time_seg = 0

for _ in range(NUM_ITERATIONS):
    start_time = time.time()
    res = segment_predict_on_image(TEST_DATA_PATH, conf=0.55)
    end_time = time.time()
    inference_time = end_time - start_time
    ncnn_total_time_seg += inference_time

ncnn_fps_det = NUM_ITERATIONS / ncnn_total_time_det
ncnn_average_time_det = ncnn_total_time_det / NUM_ITERATIONS

ncnn_fps_seg = NUM_ITERATIONS / ncnn_total_time_seg
ncnn_average_time_seg = ncnn_total_time_seg / NUM_ITERATIONS


log.warning('RESULTS')
log.info('==============PyTorch=================')
log.info(f'PyTorch Detection FPS: {pytorch_fps_det:.2f}')
log.info(f'PyTorch Detection Average Inference Time: {pytorch_average_time_det:.4f} seconds')
log.info(f'PyTorch Segmentation FPS: {pytorch_fps_seg:.2f}')
log.info(f'PyTorch Segmentation Average Inference Time: {pytorch_average_time_seg:.4f} seconds')
log.info('==============NCNN=================')
log.info(f'NCNN Detection FPS: {ncnn_fps_det:.2f}')
log.info(f'NCNN Detection Average Inference Time: {ncnn_average_time_det:.4f} seconds')
log.info(f'NCNN Segmentation FPS: {ncnn_fps_seg:.2f}')
log.info(f'NCNN Segmentation Average Inference Time: {ncnn_average_time_seg:.4f} seconds')
