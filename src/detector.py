"""Модуль детекции объектов с использованием YOLOv8."""

import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from src.config import (
    YOLO_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    IMG_SIZE,
)


class ObjectDetector:
    """Детектор объектов на основе YOLOv8 для аэросъемки с БПЛА."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = DEVICE,
        imgsz: int = IMG_SIZE,
    ):
        """Инициализация детектора и загрузка модели YOLO."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = YOLO(model_name)
        self.model.to(device)
        self.class_names = self.model.names

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Детекция объектов на изображении."""
        if image is None:
            return []

        results = self.model(image, conf=self.confidence_threshold,
                             verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                    })
        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Детекция объектов на пакете изображений."""
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
