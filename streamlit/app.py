# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_types = st.sidebar.multiselect(
    "Select Task", ['Detection', 'Segmentation'])
model_detection_path = None
model_segmentation_path = None
model_detection = None
model_segmentation = None
include_logic = False
plot_segmentation = True

if "Detection" in model_types and "Segmentation" in model_types:
    include_logic = st.sidebar.checkbox(label="Include logic")
    plot_segmentation = st.sidebar.checkbox(label="Plot segmentation")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 35)) / 100

st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# Selecting Detection Or Segmentation
if 'Detection' in model_types:
    model_detection_path = Path(settings.DETECTION_MODEL)
if 'Segmentation' in model_types:
    model_segmentation_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
if model_detection_path:
    try:
        model_detection = helper.load_model(model_detection_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_detection_path}")
        st.error(ex)

if model_segmentation_path:
    try:
        model_segmentation = helper.load_model(model_segmentation_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_segmentation_path}")
        st.error(ex)

helper_func = None

if source_radio == settings.VIDEO:
    helper_func = helper.play_stored_video

elif source_radio == settings.WEBCAM:
    helper_func = helper.play_webcam

elif source_radio == settings.RTSP:
    helper_func = helper.play_rtsp_stream

elif source_radio == settings.YOUTUBE:
    helper_func = helper.play_youtube_video

else:
    st.error("Please select a valid source type!")

if helper_func:
    helper_func(confidence, model_detection=model_detection,
                model_segmentation=model_segmentation, include_logic=include_logic, plot_segmentation=plot_segmentation)
