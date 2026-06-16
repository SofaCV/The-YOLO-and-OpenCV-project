"""Конфигурации."""
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
VSDR_IMAGES = DATA_DIR / "VisdroneFrames"
TEST_VIDEO1 = DATA_DIR / "test_video1.mp4"
TEST_VIDEO2 = DATA_DIR / "test_video2.mp4"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_FRAMES_DIR = PROCESSED_DATA_DIR / "frames"

OUTPUT_DIR = BASE_DIR / "output"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
METRICS_DIR = OUTPUT_DIR / "metrics"

DRONE_CLASSES = {
    0: "person",  # люди
    1: "bicycle",  # велосипеды
    2: "car",  # машины
    3: "motorcycle",  # мотоциклы
    4: "airplane",  # самолеты
    5: "bus",  # автобусы
    7: "truck",  # грузовики
}

YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.3
DEVICE = "cpu"

TRACKING_ALGORITHM = "CSRT"
MAX_TRACKING_FRAMES = 30
DETECT_EVERY_N_FRAMES = 5

VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

MAX_FRAMES_TO_PROCESS = None
FRAME_SKIP = 0

LOG_LEVEL = "INFO"
SAVE_RESULTS = True
