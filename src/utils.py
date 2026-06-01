"""Utility functions: image processing, logging setup, UI helpers."""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Configure application-wide logging."""
    settings = get_settings()
    logger = logging.getLogger(name or __name__.split(".")[0])

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # File handler with rotation
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    return logger


def preprocess_face(face_roi: np.ndarray,
                    target_size: tuple = (200, 200)) -> np.ndarray:
    """Resize and normalize a face region for recognition."""
    resized = cv2.resize(face_roi, target_size)
    equalized = cv2.equalizeHist(resized)
    return equalized


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if an image is blurry using Laplacian variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var() < threshold


def draw_face_ui(frame: np.ndarray, text: str,
                 status_color: tuple = (0, 255, 0)) -> np.ndarray:
    """Draw overlay UI on frame (top bar + bottom hint)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Press ESC to exit | SPACE to confirm",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    return frame


def draw_detection(frame: np.ndarray, x: int, y: int, w: int, h: int,
                   label: str, color: tuple = (0, 255, 0)) -> None:
    """Draw detection bounding box with label."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (x, y - 30),
                  (x + label_size[0] + 10, y), color, -1)
    text_color = (0, 0, 0) if color != (0, 0, 255) else (255, 255, 255)
    cv2.putText(frame, label, (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def encode_frame_jpeg(frame: np.ndarray,
                      quality: int = 85) -> bytes:
    """Encode OpenCV frame to JPEG bytes for streaming."""
    ret, buffer = cv2.imencode(".jpg", frame, [
        cv2.IMWRITE_JPEG_QUALITY, quality,
    ])
    if not ret:
        return b""
    return buffer.tobytes()
