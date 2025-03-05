import os
from pathlib import Path
import sys


# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the parent directory of the current file
root_path = os.path.dirname(current_file_path)

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(root_path)

# Get the relative path of the root directory with respect to the current working directory
ROOT = os.path.relpath(root_path, os.getcwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Camera'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = os.path.join(ROOT, '../sample_data/images')
DEFAULT_IMAGE = os.path.join(IMAGES_DIR, 'image0.jpg')
DEFAULT_DETECT_IMAGE = os.path.join(IMAGES_DIR, 'image0_detected.jpg')

# Videos config
VIDEOS_DIR = os.path.join(ROOT, '../sample_data/videos/testcase')
VIDEO_1_PATH = os.path.join(VIDEOS_DIR, 'video31.mp4')
VIDEO_2_PATH = os.path.join(VIDEOS_DIR, 'video32.mp4')
VIDEO_3_PATH = os.path.join(VIDEOS_DIR, 'video1p2.mp4')
VIDEO_4_PATH = os.path.join(VIDEOS_DIR, 'video1p3.mp4')
VIDEO_5_PATH = os.path.join(VIDEOS_DIR, 'video6p1.mp4')
# VIDEO_6_PATH = os.path.join(VIDEOS_DIR, 'case4.mp4')
# VIDEO_7_PATH = os.path.join(VIDEOS_DIR, 'case5.mp4')
# VIDEO_8_PATH = os.path.join(VIDEOS_DIR, 'case6.mp4')
# VIDEO_9_PATH = os.path.join(VIDEOS_DIR, 'test.mp4')
# VIDEO_10_PATH = os.path.join(VIDEOS_DIR, 'test.mp4')


VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
    # 'video_6': VIDEO_6_PATH,
    # 'video_7': VIDEO_7_PATH,
    # 'video_8': VIDEO_8_PATH,
    # 'video_9': VIDEO_9_PATH,
    # 'video_10': VIDEO_10_PATH,

}

# ML Model config 
MODEL_DIR = os.path.join(ROOT, 'weights')
DETECTION_MODEL_1 = os.path.join(MODEL_DIR, 'best_1n.pt')
DETECTION_MODEL_2 = os.path.join(MODEL_DIR, 'best_2s.pt')
DETECTION_MODEL_3 = os.path.join(MODEL_DIR, 'best_3m.pt')
DETECTION_MODEL_4 = os.path.join(MODEL_DIR, 'best_4l.pt')
DETECTION_MODEL_5 = os.path.join(MODEL_DIR, 'best_5x.pt')
DETECTION_MODEL_6 = os.path.join(MODEL_DIR, 'best.pt')


MODEL_DICT = {
    'yolov8n': DETECTION_MODEL_1,
    'yolov8s': DETECTION_MODEL_2,
    'yolov8m': DETECTION_MODEL_3,
    'yolov8l': DETECTION_MODEL_4,
    'yolov8x': DETECTION_MODEL_5,
    'yolov8l_new': DETECTION_MODEL_6,

}


# Detected/segmented image dirpath locator
DETECT_LOCATOR = 'detect'


# Webcam
# WEBCAM_PATH = 0 # Using IV Cam to connect camera or use default webcam
# WEBCAM_PATH = 'http://192.168.1.12:8080/video' # IP has changed each time we run IP Webcam on the phone /home
# WEBCAM_PATH = 'https://10.238.32.5:8080/video' # IP has changed each time we run IP Webcam on the phone /school
WEBCAM_PATH = 'http://192.168.43.1:8080/video' #     has changed each time we run IP Webcam on the phone /4G
