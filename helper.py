from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube

import settings
import numpy as np
from ultralytics.utils.ops import scale_image
import math


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


def display_tracker_options():
    display_tracker = st.checkbox(label="Tracker??")
    is_display_tracker = display_tracker
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, st_frame, image, is_display_tracking=None, tracker=None,
                             model_detection=None, model_segmentation=None, include_logic=False,
                             plot_segmentation=True):
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
    res_detection = None
    res_segmentation = None

    # Display object tracking, if specified
    if model_detection:
        if is_display_tracking:
            res_detection = model_detection.track(image, conf=conf, persist=True, tracker=tracker)
        else:
            # Predict the objects in the image using the YOLOv8 model
            res_detection = model_detection.predict(image, conf=conf)

    if model_segmentation:
        if is_display_tracking:
            res_segmentation = model_segmentation.track(image, conf=conf, persist=True, tracker=tracker)
        else:
            # Predict the objects in the image using the YOLOv8 model
            res_segmentation = model_segmentation.predict(image, conf=conf)

    if res_detection:
        boxes = res_detection[0].boxes.xyxy.numpy().astype("int16")
    else:
        boxes = []

    if res_segmentation and plot_segmentation:
        segmentation_mask = get_resized_mask(res_segmentation)
        segmentation_mask = merge_masks(segmentation_mask)
        if not res_segmentation[0].masks:
            segment_areas = []
        else:
            segment_areas = res_segmentation[0].masks.xy
            segment_areas = merge_polygons(segment_areas)
    else:
        segmentation_mask = np.zeros([*image.shape[:2], 1])
        segment_areas = []

    if include_logic:
        new_boxes = []
        for box in boxes:
            if not if_person_on_road(box, segmentation_mask, segment_areas):
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

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(*args,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
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
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(*args,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
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
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(*args,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
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
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            count = 0
            while vid_cap.isOpened():
                count += 1
                success, image = vid_cap.read()
                if success:
                    res = _display_detected_frames(*args,
                                                   st_frame,
                                                   image,
                                                   is_display_tracker,
                                                   tracker,
                                                   **kwargs
                                                   )
                    count_str = "0" * (4 - len(str(count))) + str(count)
                    cv2.imwrite(f'processed_images/color_img_{count_str}.png', res)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def get_resized_mask(yolo_segment_res):
    yolo_segment_res = yolo_segment_res[0]

    if yolo_segment_res.masks is None:
        return np.zeros([*yolo_segment_res.orig_shape, 1])

    result_mask = yolo_segment_res.masks.data.numpy()
    result_mask = np.moveaxis(result_mask, 0, -1)
    result_mask = scale_image(result_mask, yolo_segment_res.orig_shape)

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


def if_person_on_road(bbox_xyxy, segment_mask, segment_polygons, threshold=2):
    x_mean = (bbox_xyxy[2] + bbox_xyxy[0]) // 2
    y_bot = bbox_xyxy[3]

    if segment_mask[y_bot][x_mean] == 1:
        return True

    min_dist = float("+Inf")

    for x, y in segment_polygons:
        dist = math.sqrt((x_mean - x) ** 2 + (y_bot - y) ** 2)
        min_dist = min(min_dist, dist)

    return min_dist < threshold


def merge_masks(mask):
    result_mask = np.any(mask, axis=2)
    return np.expand_dims(result_mask, axis=2)


def merge_polygons(segment_polygons):
    return np.concatenate(segment_polygons)
