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
detection = st.sidebar.checkbox(label="Detection")
segmentation = st.sidebar.checkbox(label="Segmentation")
include_logic = False
plot_segmentation = True

if detection and segmentation:
    include_logic = st.sidebar.checkbox(label="Include logic")
    plot_segmentation = st.sidebar.checkbox(label="Plot segmentation")


st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

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
    helper_func(detection=detection, segmentation=segmentation, plot_segmentation=plot_segmentation,
                include_logic=include_logic)
