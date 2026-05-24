# Чтение датасета
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import sys
import logging

from src.config import (
    BASE_DIR,
    DATA_DIR,
    VSDR_IMAGES,
    VSDR_ANNOT,
    TEST_VIDEO,
    PROCESSED_DATA_DIR,
    PROCESSED_FRAMES_DIR,
    TARGET_CLASSES,
    LOG_LEVEL
)


class VisDroneDataset:
    def __init__(self, images_dir=None, annotations_file=None):
        self.images_dir = images_dir if images_dir else VSDR_IMAGES
        self.annot_file = annotations_file if annotations_file else VSDR_ANNOT
        self.annotations: Dict[int, List[Dict]] = {}
        self.image_files: List[Path] = []


class VideoReader:
    def __init__(self, video_path=None):
        self.video_path = video_path if video_path else TEST_VIDEO
