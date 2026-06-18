"""Модуль визуализации результатов детекции и трекинга."""

import cv2
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


class Visualizer:
    """Визуализация bounding boxes и метрик."""

    def __init__(
        self,
        show_confidence: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """Инициализация визуализатора."""
        self.show_confidence = show_confidence
        self.line_thickness = line_thickness
        self.font_scale = font_scale

    def draw_detections(
        self,
        image: np.ndarray,
        tracking_info: Dict[int, Dict],
    ) -> np.ndarray:
        """Отрисовка треков на изображении."""
        if image is None:
            return None

        result = image.copy()

        for track_id, track in tracking_info.items():
            bbox = track.get("bbox")
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            class_name = track.get("class_name", "unknown")
            confidence = track.get("confidence", 0.0)
            color = track.get("color", (255, 255, 255))

            label = f"ID:{track_id}"
            if class_name != "unknown":
                label += f" {class_name}"
            if self.show_confidence and confidence > 0:
                label += f" {confidence:.2f}"

            result = self._draw_bbox(result, (x1, y1, x2, y2), color, label)

        return result

    def _draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        label: str = "",
    ) -> np.ndarray:
        """Отрисовка одной bounding box с подписью."""
        x1, y1, x2, y2 = bbox
        result = image.copy()

        cv2.rectangle(result, (x1, y1), (x2, y2), color, self.line_thickness)

        if label:
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )

            cv2.rectangle(
                result,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1,
            )

            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )

        return result

    def draw_metrics(
        self,
        image: np.ndarray,
        fps: float,
        num_objects: int,
        frame_id: int,
    ) -> np.ndarray:
        """Отрисовка метрик на изображении."""
        if image is None:
            return None

        result = image.copy()

        overlay = result.copy()
        cv2.rectangle(overlay, (5, 5), (280, 100), (0, 0, 0), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        y = 25
        cv2.putText(
            result,
            f"Frame: {frame_id}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        y += 22
        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        y += 22
        cv2.putText(
            result,
            f"Objects: {num_objects}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        return result

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: Path,
        frame_id: int,
    ) -> None:
        """Сохранение кадра в файл."""
        if frame is None:
            return

        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"frame_{frame_id:06d}.jpg"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), frame_bgr)
