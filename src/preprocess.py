"""Модуль предобработки изображений с БПЛА."""

import cv2
import numpy as np
from typing import Tuple


def __apply_clahe_if_needed(
    frame: np.ndarray,
    std_threshold: float = 35.0,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Повышение контрастности изображения при необходимости."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if gray.std() < std_threshold:
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        clahe = cv2.createCLAHE(clip_limit, grid_size)
        lightness = clahe.apply(lightness)
        lab = cv2.merge((lightness, green_red, blue_yellow))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return frame


def __apply_gamma_if_needed(
    frame: np.ndarray,
    low_threshold: float = 70.0,
    high_threshold: float = 200.0,
    gamma_low: float = 1.8,
    gamma_high: float = 0.7,
) -> np.ndarray:
    """Коррекция гаммы для слишком темных или светлых изображений."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean_bright = gray.mean()

    if mean_bright < low_threshold:
        gamma = gamma_low
    elif mean_bright > high_threshold:
        gamma = gamma_high
    else:
        return frame

    inv_gamma = 1.0 / gamma

    table_list = []
    for i in range(256):
        step1 = i / 255.0
        step2 = step1**inv_gamma
        step3 = step2 * 255
        table_list.append(step3)

    table_float = np.array(table_list)
    table = table_float.astype(np.uint8)

    return cv2.LUT(frame, table)


def preprocess_frame(
    frame: np.ndarray,
    enable_clahe: bool = True,
    enable_gamma: bool = True,
    clahe_std_threshold: float = 35.0,
    low_threshold: float = 70.0,
    high_threshold: float = 200.0,
) -> np.ndarray:
    """Основная функция предобработки кадра."""
    if frame is None:
        return None

    if enable_clahe:
        frame = __apply_clahe_if_needed(frame, clahe_std_threshold)

    if enable_gamma:
        frame = __apply_gamma_if_needed(frame, low_threshold, high_threshold)

    return frame
