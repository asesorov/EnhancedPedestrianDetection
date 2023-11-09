from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube

import settings
import numpy as np
from ultralytics.utils.ops import scale_image
import math
import io
from src.inference.inference_yolo_ncnn import detect_get_bounding_boxes, segment_predict_on_image


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path: The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def _display_detected_frames(st_frame, image, detection=False, segmentation=False,
                             plot_segmentation=True, include_logic=False):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    boxes = []
    if detection:
        boxes = detect_get_bounding_boxes(image)

    res_segmentation = None
    if segmentation:
        res_segmentation = segment_predict_on_image(image, conf=0.70)

    if not res_segmentation is None and plot_segmentation:
        segmentation_mask = get_resized_mask(res_segmentation, image.shape)
        segmentation_mask = merge_masks(segmentation_mask)
    else:
        segmentation_mask = np.zeros([*image.shape[:2], 1])

    if include_logic:
        new_boxes = []
        for box in boxes:
            if not if_person_on_road(box, segmentation_mask):
                continue
            new_boxes.append(box)
        boxes = new_boxes

    # # Plot the detected objects on the video frame
    res_plotted = plot_detect_segment(image, boxes, segmentation_mask)
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    return res_plotted


def play_youtube_video(*args, **kwargs):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(st_frame,
                                             image,
                                             *args,
                                             **kwargs
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(*args, **kwargs):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(st_frame,
                                             image,
                                             *args,
                                             **kwargs
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(*args, **kwargs):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(st_frame,
                                             image,
                                             *args,
                                             **kwargs
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(*args, **kwargs):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    # source_vid = st.sidebar.selectbox(
    #     "Choose a video...", settings.VIDEOS_DICT.keys())
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_file:
        video_bytes = io.BytesIO(uploaded_file.read())
        temporary_location = "videos/video"

        with open(temporary_location, 'wb') as out:
            out.write(video_bytes.read())

        out.close()

    if uploaded_file and video_bytes:
        st.video(video_bytes)

    # with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
    #     video_bytes = video_file.read()

    if st.sidebar.button('Detect Video Objects') and uploaded_file:
        try:
            vid_cap = cv2.VideoCapture(temporary_location)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(st_frame,
                                             image,
                                             *args,
                                             **kwargs
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def get_resized_mask(yolo_segment_res, img_shape):
    result_mask = np.moveaxis(yolo_segment_res, 0, -1)
    result_mask = scale_image(result_mask, [*img_shape[:2], 1])

    return result_mask


def draw_mask(image, mask_generated):
    masked_image = image.copy()

    masked_image = np.where(mask_generated.astype(int),
                            np.array([255, 0, 0], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.7, masked_image, 0.3, 0)


def plot_detect_segment(image, bboxs_xyxy=[], segment_mask=None):
    res_image = image.copy()

    for bbox in bboxs_xyxy:
        cv2.rectangle(res_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

    if segment_mask is None:
        return res_image

    return draw_mask(res_image, segment_mask)


def if_person_on_road(bbox_xyxy, segment_mask):
    x_mean = (bbox_xyxy[2] + bbox_xyxy[0]) // 2
    y_bot = bbox_xyxy[3]

    if segment_mask[y_bot][x_mean] == 1:
        return True

    return False


def merge_masks(mask):
    result_mask = np.any(mask, axis=2)
    return np.expand_dims(result_mask, axis=2)

