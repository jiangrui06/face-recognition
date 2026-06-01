"""Face detection module - Strategy pattern with Haar Cascade and OpenCV DNN.

Usage::

    detector = create_detector("haar")
    faces = detector.detect(gray_frame)
"""

import logging
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)


class FaceDetector(ABC):
    """Abstract base class for face detectors."""

    @abstractmethod
    def detect(self, gray_frame: np.ndarray,
               min_size: int = 80) -> list[tuple[int, int, int, int]]:
        """Detect faces in a grayscale image.

        Returns list of (x, y, w, h) bounding boxes.
        """


class HaarCascadeDetector(FaceDetector):
    """Haar Cascade-based face detector.

    Fast and works well for frontal faces. Falls back to profile
    cascade when no frontal face is found.
    """

    def __init__(self):
        self.frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    def detect(self, gray_frame: np.ndarray,
               min_size: int = 80) -> list[tuple[int, int, int, int]]:
        faces = self.frontal_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_size, min_size),
        )
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5,
                minSize=(min_size, min_size),
            )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


class DNNFaceDetector(FaceDetector):
    """OpenCV DNN-based face detector using Caffe SSD model.

    More accurate than Haar, especially for non-frontal faces and
    challenging lighting conditions. Downloads model files on first use.
    """

    MODEL_URL = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                 "dnn_samples_face_detector_20170830/"
                 "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    CONFIG_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                  "samples/dnn/face_detector/deploy.prototxt")

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        model_path, config_path = self._ensure_models()
        if model_path and config_path:
            self.net = cv2.dnn.readNetFromCaffe(str(config_path),
                                                str(model_path))
        else:
            logger.warning("DNN model files not available, DNN detector disabled")
            self.net = None

    def _ensure_models(self) -> tuple[Optional[Path], Optional[Path]]:
        """Download model files if not present."""
        cache_dir = Path(get_settings().model_cache_dir)
        os.makedirs(str(cache_dir), exist_ok=True)

        model_path = cache_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_path = cache_dir / "deploy.prototxt"

        if not model_path.exists():
            logger.info("Downloading DNN face detector model...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, str(model_path))
                logger.info("Model downloaded: %s", model_path)
            except Exception as e:
                logger.error("Failed to download model: %s", e)
                return None, None

        if not config_path.exists():
            logger.info("Downloading DNN face detector config...")
            try:
                urllib.request.urlretrieve(self.CONFIG_URL, str(config_path))
                logger.info("Config downloaded: %s", config_path)
            except Exception as e:
                logger.error("Failed to download config: %s", e)
                return model_path, None

        return model_path, config_path

    def detect(self, gray_frame: np.ndarray,
               min_size: int = 80) -> list[tuple[int, int, int, int]]:
        if self.net is None:
            return []

        h, w = gray_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            gray_frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw < min_size or fh < min_size:
                continue
            faces.append((x1, y1, fw, fh))

        return faces


def create_detector(detector_type: str = "haar") -> FaceDetector:
    """Factory function for creating face detectors.

    Args:
        detector_type: "haar" for Haar Cascade, "dnn" for DNN-based.

    Returns:
        FaceDetector instance. Falls back to Haar if DNN is unavailable.
    """
    if detector_type.lower() == "dnn":
        detector = DNNFaceDetector()
        if detector.net is None:
            logger.warning("DNN detector unavailable, falling back to Haar")
            return HaarCascadeDetector()
        return detector
    return HaarCascadeDetector()
