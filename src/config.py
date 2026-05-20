from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
VISDRONE_IMAGES = DATA_DIR / "VisdroneFrames"
VISDRONE_ANNOTATIONS = DATA_DIR / "markup"
TEST_VIDEO = DATA_DIR / "test_video"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_FRAMES_DIR = PROCESSED_DATA_DIR / "frames"

OUTPUT_DIR = BASE_DIR / "output"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
METRICS_DIR = OUTPUT_DIR / "metrics"


YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = "cuda"

# person, bicycle, car, motorcycle, airplane, bus, train, truck, traffic light.
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 9]

COLORS = {
    "person": (0, 255, 0),
    "car": (255, 0, 0),
    "truck": (0, 0, 255),
    "bus": (0, 255, 255),
    "motorcycle": (255, 255, 0),
    "bicycle": (255, 0, 255),
    "airplane": (0, 69, 255),
    "train": (128, 128, 128),
    "traffic light": (147, 20, 255),
}

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
