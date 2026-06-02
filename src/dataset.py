# Чтение датасета
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from src.config import VSDR_IMAGES, VSDR_ANNOT, TEST_VIDEO


class ImageReader:  # Чтение фото
    def __init__(self, images_dir=VSDR_IMAGES):
        self.images_dir = images_dir if images_dir else VSDR_IMAGES
        self.image_files: List[Path] = []
        self._load_image_list()

    def _load_image_list(self):
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = sorted(self.images_dir.glob("*.png"))

    def get_image(self, frame_id: int) -> Optional[np.ndarray]:
        if frame_id >= len(self.image_files):
            return None
        image_path = self.image_files[frame_id]
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class VideoReader:  # Чтение видео
    def __init__(self, video_path=None):
        self.video_path = video_path if video_path else TEST_VIDEO
        self.cap = None
        self._frames_dir = None
        self._image_reader = None

    def _open_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.video_path}")
        return self.cap

    def __len__(self) -> int:
        cap = self._open_video()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.cap = None
        return count

    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        cap = self._open_video()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_generator(self) -> Generator[
                                               Tuple[int, np.ndarray],
                                               None,
                                               None
                                            ]:
        cap = self._open_video()
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                yield frame_id, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_id += 1
        finally:
            cap.release()
            self.cap = None

    def __del__(self):
        if self.cap:
            self.cap.release()


class AnnotationsReader:
    def __init__(self, annotations_file=None):
        self.annot_file = annotations_file if annotations_file else VSDR_ANNOT
        self.annotations: Dict[int, List[Dict]] = {}
        self._load_annotations()

    def _load_annotations(self):
        if not self.annot_file.exists():
            print(f"Файл аннотаций не найден: {self.annot_file}")
            return

        with open(self.annot_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = list(map(int, line.split(",")))
                    if len(parts) < 8:
                        continue

                    frame_id = parts[0]
                    x, y, w, h = parts[1:5]
                    confidence = parts[5]
                    class_id = parts[6]

                    if confidence == 0:
                        continue

                    if frame_id not in self.annotations:
                        self.annotations[frame_id] = []

                    self.annotations[frame_id].append(
                        {"bbox": [x, y, w, h], "class_id": class_id}
                    )

                except ValueError:
                    print(f"Ошибка в строке {line_num}: {line}")

    def get_annotations(self, frame_id: int) -> List[Dict]:
        return self.annotations.get(frame_id, [])

    def get_frames_with_annotations(self) -> List[int]:
        return list(self.annotations.keys())

    def get_all_annotations(self) -> Dict[int, List[Dict]]:
        return self.annotations

    def __len__(self) -> int:
        return len(self.annotations)
