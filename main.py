import cv2
import time
from pathlib import Path
from typing import Optional, List

from src.detector import ObjectDetector
from src.track import ByteTrackTracker
from src.visualizer import Visualizer
from src.dataset import VideoReader
from src.config import (
    OUTPUT_DIR,
    TEST_VIDEO,
    VSDR_IMAGES,
    MAX_FRAMES_TO_PROCESS,
    IMG_SIZE,
)


class SimpleDroneProcessor:
    """Упрощенный процессор для обработки видео/изображений."""

    def __init__(self, imgsz=IMG_SIZE):
        self.detector = ObjectDetector(imgsz=imgsz)
        self.tracker = ByteTrackTracker()
        self.visualizer = Visualizer()
        self.processed_count = 0
        self.total_objects = 0

    def process_video(self, video_path: Path, output_path: Optional[Path] = None):
        """Обработка видео с БПЛА."""
        print(f"\n📹 Обработка видео: {video_path.name}")

        if output_path is None:
            output_path = OUTPUT_DIR / "videos" / f"{video_path.stem}_output.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        reader = VideoReader(video_path)
        total_frames = len(reader)
        if MAX_FRAMES_TO_PROCESS:
            total_frames = min(total_frames, MAX_FRAMES_TO_PROCESS)

        print(f"Всего кадров: {total_frames}")

        first_frame = reader.get_frame(0)
        if first_frame is None:
            print(" Не удалось прочитать видео")
            return

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, 30, (w, h))

        start_time = time.time()
        frame_count = 0
        self.processed_count = 0
        self.total_objects = 0

        for frame_id, frame in reader.get_frame_generator():
            if MAX_FRAMES_TO_PROCESS and frame_id >= MAX_FRAMES_TO_PROCESS:
                break

            detections = self.detector.detect(frame)
            tracked = self.tracker.update(detections)

            result = self.visualizer.draw_detections(frame, tracked)
            fps = self._calculate_fps(start_time, frame_count + 1)
            result = self.visualizer.draw_metrics(result, fps, len(tracked),
                                                  frame_id)

            writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

            self.processed_count += 1
            self.total_objects += len(tracked)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"  Прогресс: {frame_count}/{total_frames} кадров")

        writer.release()
        total_time = time.time() - start_time

        print(f"  - Обнаружено объектов: {self.total_objects}")
        print(f"  - Время: {total_time:.2f} сек")
        print(f"  - Результат сохранен: {output_path}")

    def process_images(self, images_dir: Path,
                       output_dir: Optional[Path] = None):
        """Обработка папки с изображениями."""

        print(f"\n Обработка папки: {images_dir.name}")

        if output_dir is None:
            output_dir = OUTPUT_DIR / "processed_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files: List = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        image_files = sorted(image_files)

        if not image_files:
            print(f" Изображения не найдены в {images_dir}")
            return

        print(f"Найдено изображений: {len(image_files)}")

        start_time = time.time()
        self.processed_count = 0
        self.total_objects = 0

        for idx, img_path in enumerate(image_files, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = self.detector.detect(frame)
            tracked = self.tracker.update(detections)

            result = self.visualizer.draw_detections(frame, tracked)

            output_path = output_dir / f"{img_path.stem}_output{img_path.suffix}"
            cv2.imwrite(str(output_path), cv2.cvtColor(result,
                                                       cv2.COLOR_RGB2BGR))

            self.processed_count += 1
            self.total_objects += len(tracked)

            if idx % 10 == 0 or idx == len(image_files):
                print(f"  Прогресс: {idx}/{len(image_files)} изображений")

        total_time = time.time() - start_time

        print(f"  - Обнаружено объектов: {self.total_objects}")
        print(f"  - Время: {total_time:.2f} сек")
        print(f"  - Результаты сохранены в: {output_dir}")

    def _calculate_fps(self, start_time: float, frames: int) -> float:
        """Расчет FPS."""
        elapsed = time.time() - start_time
        return frames / elapsed if elapsed > 0 else 0


def main():
    """Главная функция с простым меню."""

    print("\nВыберите режим:")
    print("  1 - Обработать видео")
    print("  2 - Обработать папку с изображениями")

    choice = input("\nВаш выбор (0-2): ").strip()

    processor = SimpleDroneProcessor()

    if choice == "1":
        use_default = (
            input("\nИспользовать тестовое видео по умолчанию? (y/n): ").strip().lower()
        )

        if use_default in ["y", "yes", "д", "да", ""]:
            video_path = TEST_VIDEO
        else:
            video_path = Path(input("Введите путь к видео: ").strip())

        if not video_path.exists():
            print(f"❌ Видео не найдено: {video_path}")
            return

        save_path = input("\nПуть для сохранения (Enter для default): ").strip()
        processor.process_video(video_path, Path(save_path) if save_path else None)

    elif choice == "2":
        use_default = (
            input("\nИспользовать тестовую папку? (y/n): ").strip().lower()
        )

        if use_default in ["y", "yes", "д", "да", ""]:
            images_dir = VSDR_IMAGES
        else:
            images_dir = Path(input("Введите путь к папке с изображениями: ").strip())

        if not images_dir.exists():
            print(f"❌ Папка не найдена: {images_dir}")
            return

        save_dir = input("\nПуть для сохранения (Enter для default): ").strip()
        processor.process_images(images_dir, Path(save_dir) if save_dir else None)

    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    main()
