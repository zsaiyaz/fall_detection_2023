from ultralytics import YOLO
import streamlit as st
import cv2
import os
import settings
import tracker
import time

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_option():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    return is_display_tracker


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None):
    image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image, conf=conf)
    result_tensor = res[0].boxes
    if is_display_tracking:
        tracker._display_detected_tracks(result_tensor.data, image)

    # Plot after drawing the tracking
    res_plotted = res[0].plot()
    # st.write(dir(res))
    # st_frame.image(str(fps))
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True,
                                      )

def play_webcam(conf, model):
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
    is_display_tracker = display_tracker_option()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if cv2.waitKey(1) == ord('q'):
                    break
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
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

    # source_vid = st.sidebar.file_uploader(
    #     "Choose a video...", type=("mp4"))

    is_display_tracker = display_tracker_option()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
           
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker
                                             )
                    # cv2.waitKey(5000)
                    # fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                    # st.sidebar.write("FPS: " + str(fps))
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

# def save_video(conf, model):

#     save_radio = st.sidebar.radio("Save video to download", ["Yes", "No"])
#     save = True if save_radio == 'Yes' else False
#     res = model.predict(settings.VIDEOS_DICT.get(source_vid),
#                                         save=False,
#                                         save_txt=False,
#                                         exist_ok=True,
#                                         conf=confidence
#                                         )

if __name__ == "__main__":
    main()