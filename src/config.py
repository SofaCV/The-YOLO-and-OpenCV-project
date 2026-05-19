from pathlib import Path


BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = BASE_DIR / "data"
VISDRONE_IMAGES_ONE_DIR = RAW_DATA_DIR / "sequences" / "uav0000086_00000_v"
VISDRONE_IMAGES_TWO_DIR = RAW_DATA_DIR / "sequences" / "uav0000117_02622_v"
VISDRONE_ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations" / ""
VISDRONE_ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations" / ""

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_FRAMES_DIR = PROCESSED_DATA_DIR / "frames"


TEST_DATA_DIR = BASE_DIR / "test_video"
TEST_VIDEO_PATH = TEST_DATA_DIR / "тестовое видео 1.mp4"


OUTPUT_DIR = BASE_DIR / "output"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"


YOLO_MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = "cuda"  # или "cpu"

# person, bicycle, car, motorcycle, airplane, bus, train, truck, traffic light 
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 9]  

TRACKING_ALGORITHM = "CSRT"
MAX_TRACKING_FRAMES = 30

VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

LOG_LEVEL = "INFO"
SAVE_RESULTS = True
