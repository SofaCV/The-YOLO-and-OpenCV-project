"""Модуль трекинга объектов с использованием ByteTrack."""

from typing import List, Dict, Tuple
from collections import deque
import colorsys

from src.config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_TRACKING_FRAMES


class ByteTrackTracker:
    """Трекер объектов на основе ByteTrack."""

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        max_lost_frames: int = MAX_TRACKING_FRAMES,
        track_history_length: int = 30,
    ):
        """Инициализация ByteTrack трекера."""
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.track_history_length = track_history_length

        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 0

        self.track_history: Dict[int, deque] = {}

        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _generate_color(self, track_id: int) -> Tuple[int, int, int]:
        """Генерирует уникальный цвет для трека на основе его ID."""
        if track_id not in self._color_cache:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self._color_cache[track_id] = (
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255),
            )
        return self._color_cache[track_id]

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Вычисление IoU между двумя bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        return inter_area / (bbox1_area + bbox2_area - inter_area)

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Обновление трекера на основе новых детекций."""
        high_conf_detections = [
            d for d in detections if d["confidence"] >= self.confidence_threshold
        ]
        low_conf_detections = [
            d for d in detections if d["confidence"] < self.confidence_threshold
        ]

        matched_tracks = set()
        used_detections = set()
        for det_idx, detection in enumerate(high_conf_detections):
            best_iou: float = 0.0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue

                iou = self._calculate_iou(detection["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                self.tracks[best_track_id].update(
                    {
                        "bbox": detection["bbox"],
                        "confidence": detection["confidence"],
                        "class_id": detection["class_id"],
                        "class_name": detection["class_name"],
                        "lost_frames": 0,
                    }
                )
                matched_tracks.add(best_track_id)
                used_detections.add(det_idx)

        for det_idx, detection in enumerate(low_conf_detections):
            if det_idx in used_detections:
                continue

            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue

                iou = self._calculate_iou(detection["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold * 0.7:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                self.tracks[best_track_id].update(
                    {
                        "bbox": detection["bbox"],
                        "confidence": detection["confidence"],
                        "lost_frames": 0,
                    }
                )
                matched_tracks.add(best_track_id)

        for det_idx, detection in enumerate(high_conf_detections):
            if det_idx not in used_detections:
                track_id = self.next_track_id
                self.tracks[track_id] = {
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "color": self._generate_color(track_id),
                    "lost_frames": 0,
                }
                self.track_history[track_id] = deque(maxlen=self.track_history_length)
                self.next_track_id += 1

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["lost_frames"] += 1

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                del self.tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]

        for track_id, track in self.tracks.items():
            bbox = track["bbox"]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.track_history_length)
            self.track_history[track_id].append((center_x, center_y))

        result = {}
        for track_id, track in self.tracks.items():
            result[track_id] = track.copy()
            result[track_id]["trajectory"] = list(self.track_history.get(track_id, []))

        return result

    def get_active_tracks(self) -> Dict[int, Dict]:
        """Возвращает все активные треки (без обновления)."""
        result = {}
        for track_id, track in self.tracks.items():
            result[track_id] = track.copy()
            result[track_id]["trajectory"] = list(self.track_history.get(track_id, []))
        return result

    def clear(self) -> None:
        """Очищает все треки и историю."""
        self.tracks.clear()
        self.track_history.clear()
        self._color_cache.clear()
        self.next_track_id = 0
