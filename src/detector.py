"""Модуль детекции объектов с использованием YOLOv8."""

import numpy as np
import hashlib
from ultralytics import YOLO
from typing import List, Dict, Tuple
from src.config import (
    YOLO_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
)


class ObjectDetector:
    """Детектор объектов на основе YOLOv8 для аэросъемки с БПЛА."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: str = DEVICE,
    ):
        """Инициализация детектора и загрузка модели YOLO."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_name)
        self.model.to(device)
        self.class_names = self.model.names
        self._color_cache: Dict[str, Tuple[int, int, int]] = {}

    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Генерация уникального цвета для класса (с кэшированием)."""
        if class_name is None:
            return (255, 255, 255)

        if class_name in self._color_cache:
            return self._color_cache[class_name]

        hash_obj = hashlib.md5(class_name.encode())
        hash_hex = hash_obj.hexdigest()

        r = int(hash_hex[0:2], 16)
        g = int(hash_hex[2:4], 16)
        b = int(hash_hex[4:6], 16)

        color = (r, g, b)
        self._color_cache[class_name] = color

        return color

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Детекция объектов на изображении."""
        if image is None:
            return []

        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id not in self.class_names:
                    continue

                class_name = self.class_names.get(class_id)
                if class_name is None:
                    class_name = f"class_{class_id}"

                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                    "color": self._get_color(class_name),
                }
                detections.append(detection)

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Детекция объектов на пакете изображений."""
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
