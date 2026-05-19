import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class VisDroneLoader:
    def __init__(self, images_dir: Path, annotations_dir: Path):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.jpeg")) +
                                  list(self.images_dir.glob("*.png")))
        
        print(f"📸 Загружено {len(self.image_paths)} изображений из {self.images_dir.name}")
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List, str]:
        """
        Получить изображение и аннотацию по индексу
        
        Returns:
            image: изображение в формате RGB (numpy array)
            boxes: список боксов в формате [x1, y1, x2, y2, class_id]
            image_id: имя файла (без расширения)
        """
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ищем аннотацию
        annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
        boxes = self._load_annotations(annotation_path)
        
        return image, boxes, image_path.stem
    
    def _load_annotations(self, annotation_path: Path) -> List:
        """
        Загружает аннотации из TXT файла (формат VisDrone)
        
        Формат VisDrone:
        <x1>,<y1>,<x2>,<y2>,<confidence>,<class>,<truncation>,<occlusion>
        """
        boxes = []
        
        if not annotation_path.exists():
            return boxes
        
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    x2 = int(parts[2])
                    y2 = int(parts[3])
                    class_id = int(parts[5])
                    
                    # Фильтруем только нужные классы (если заданы)
                    if config.TARGET_CLASSES and class_id not in config.TARGET_CLASSES:
                        continue
                    
                    boxes.append([x1, y1, x2, y2, class_id])
        
        return boxes
    
    def get_frame_generator(self) -> Generator:
        """Генератор для последовательного чтения кадров (экономит память)"""
        for idx in range(len(self)):
            image, boxes, img_id = self[idx]
            yield image, boxes, img_id


class MultiVisDroneLoader:
    """Загрузчик для нескольких папок VisDrone сразу"""
    
    def __init__(self):
        self.loaders = []
        
        # Загружаем все пары (images, annotations) из конфига
        for img_dir, ann_dir in zip(config.VISDRONE_IMAGE_DIRS, 
                                     config.VISDRONE_ANNOTATION_DIRS):
            if img_dir.exists() and ann_dir.exists():
                loader = VisDroneLoader(img_dir, ann_dir)
                self.loaders.append(loader)
                print(f"  ✅ {img_dir.name} -> {len(loader)} кадров")
            else:
                print(f"  ⚠️ Пропущена папка: {img_dir}")
        
        self.total_frames = sum(len(l) for l in self.loaders)
        print(f"\n📊 Всего загружено: {self.total_frames} кадров")
    
    def __len__(self) -> int:
        return self.total_frames
    
    def get_all_frames(self) -> Generator:
        """Генератор для последовательного чтения из всех папок"""
        for loader in self.loaders:
            for image, boxes, img_id in loader.get_frame_generator():
                yield image, boxes, img_id


# ============================================================================
# 2. ЗАГРУЗЧИК ВИДЕО
# ============================================================================

class VideoLoader:
    """
    Загрузчик видео файлов
    Работает с вашим test_video/тестовое видео 1.mp4
    """
    
    def __init__(self, video_path: Union[str, Path], 
                 target_size: Tuple[int, int] = None,
                 skip_frames: int = 0,
                 max_frames: int = None):
        """
        Args:
            video_path: путь к видео файлу
            target_size: (width, height) для ресайза (None = без изменений)
            skip_frames: сколько кадров пропускать (0 = все кадры)
            max_frames: максимальное количество кадров для чтения
        """
        self.video_path = Path(video_path)
        self.target_size = target_size
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
        self._open_video()
    
    def _open_video(self):
        """Открывает видео и получает информацию"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Видео не найдено: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎬 Видео загружено: {self.video_path.name}")
        print(f"   Размер: {self.width}x{self.height}, FPS: {self.fps}")
        print(f"   Всего кадров: {self.total_frames}")
    
    def __len__(self) -> int:
        if self.max_frames:
            return min(self.max_frames, self.total_frames)
        return self.total_frames
    
    def __iter__(self):
        """Итератор для покадрового чтения"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Начало видео
        frame_count = 0
        frames_processed = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Пропуск кадров
            if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Ограничение по количеству кадров
            if self.max_frames and frames_processed >= self.max_frames:
                break
            
            # Конвертируем BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ресайз
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            
            yield frame, frame_count
            frame_count += 1
            frames_processed += 1
    
    def get_frame_by_index(self, idx: int) -> Optional[np.ndarray]:
        """Получить конкретный кадр по индексу"""
        if not self.cap:
            self._open_video()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.target_size:
                frame = cv2.resize(frame, self.target_size)
            return frame
        return None
    
    def close(self):
        """Закрыть видео"""
        if self.cap:
            self.cap.release()


# ============================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def extract_frames_from_video(video_path: Path, 
                              output_dir: Path,
                              skip_frames: int = 0,
                              max_frames: int = None) -> List[Path]:
    """
    Извлечение кадров из видео и сохранение как JPEG
    
    Args:
        video_path: путь к видео
        output_dir: папка для сохранения кадров
        skip_frames: пропускать кадры
        max_frames: максимум кадров
    
    Returns:
        список путей к сохранённым кадрам
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = VideoLoader(video_path, skip_frames=skip_frames, max_frames=max_frames)
    saved_paths = []
    
    for frame, frame_idx in loader:
        # Сохраняем кадр
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame_bgr)
        saved_paths.append(frame_path)
        
        # Прогресс
        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"   Извлечено кадров: {frame_idx}")
    
    loader.close()
    print(f"✅ Сохранено {len(saved_paths)} кадров в {output_dir}")
    return saved_paths


# ============================================================================
# 4. ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================================================

def demo_visdrone():
    """Демонстрация загрузки VisDrone"""
    print("\n" + "="*50)
    print("📸 ДЕМО: Загрузка VisDrone")
    print("="*50)
    
    # Используем первую папку из конфига
    if config.VISDRONE_IMAGE_DIRS:
        loader = VisDroneLoader(
            config.VISDRONE_IMAGE_DIRS[0],
            config.VISDRONE_ANNOTATION_DIRS[0]
        )
        
        # Загружаем первые 5 кадров
        for i in range(min(5, len(loader))):
            image, boxes, img_id = loader[i]
            print(f"  Кадр {img_id}: {image.shape}, найдено объектов: {len(boxes)}")
        
        return loader
    else:
        print("  ⚠️ Папки VisDrone не найдены в конфиге")
        return None


def demo_video():
    """Демонстрация загрузки видео"""
    print("\n" + "="*50)
    print("🎬 ДЕМО: Загрузка видео")
    print("="*50)
    
    video_path = config.TEST_VIDEO_PATH
    
    if not video_path.exists():
        print(f"  ⚠️ Видео не найдено: {video_path}")
        return None
    
    # Создаём загрузчик
    loader = VideoLoader(video_path, target_size=(640, 480), max_frames=10)
    
    # Проходим по кадрам
    for i, (frame, frame_idx) in enumerate(loader):
        print(f"  Кадр {frame_idx}: размер {frame.shape}")
    
    loader.close()
    return loader


def demo_multi_visdrone():
    """Демонстрация загрузки из нескольких папок"""
    print("\n" + "="*50)
    print("📸 ДЕМО: Мульти-загрузчик VisDrone")
    print("="*50)
    
    loader = MultiVisDroneLoader()
    
    # Показываем первые 5 кадров из всех папок
    count = 0
    for image, boxes, img_id in loader.get_all_frames():
        print(f"  {img_id}: {image.shape}, объектов: {len(boxes)}")
        count += 1
        if count >= 5:
            break
    
    return loader


# ============================================================================
# 5. ЗАПУСК ПРИМЕРОВ
# ============================================================================

if __name__ == "__main__":
    # Создаём папки
    config.ensure_dirs()
    
    # Демонстрация
    print("\n🔧 ЗАГРУЗЧИК ДАННЫХ ДЛЯ БПЛА")
    print("="*50)
    
    # VisDrone
    visdrone_loader = demo_visdrone()
    
    # Видео
    video_loader = demo_video()
    
    # Мульти-загрузчик
    multi_loader = demo_multi_visdrone()
    
    print("\n✅ Все демо завершены")