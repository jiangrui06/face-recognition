"""Face recognition module - LBPH (Local Binary Patterns Histograms).

Handles model training and prediction. Persists the trained model to disk.
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings
from src.database import DatabaseManager
from src.exceptions import ModelNotFoundError, TrainingError

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """LBPH-based face recognizer with database integration."""

    def __init__(self):
        self.settings = get_settings()
        self.db = DatabaseManager()
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8,
        )
        self._trained = False
        self._name_cache: dict[int, str] = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> int:
        """Train the recognizer on all face samples from the database.

        Returns:
            Number of training samples used.

        Raises:
            TrainingError: If not enough samples are available.
        """
        face_samples = []
        labels = []

        with self.db.get_connection() as conn:
            employees = self.db.get_all_employees(conn)
            if not employees:
                raise TrainingError("没有已注册的员工，请先注册")

            for emp in employees:
                emp_id = emp["id"]
                samples = self.db.get_face_samples(conn, emp_id)
                for sample in samples:
                    img = cv2.imread(sample["image_path"],
                                     cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    face_samples.append(img)
                    labels.append(emp_id)
                self._name_cache[emp_id] = emp["name"]

        if len(face_samples) < 10:
            raise TrainingError(
                f"样本不足: 当前 {len(face_samples)} 张, 至少需要 10 张"
            )

        logger.info("Training on %d samples, %d employees",
                    len(face_samples), len(set(labels)))
        self.model.train(face_samples, np.array(labels, dtype=np.int32))
        self.model.write(self.settings.trainer_file)
        self._trained = True
        logger.info("Model saved to %s", self.settings.trainer_file)
        return len(face_samples)

    def load_model(self) -> bool:
        """Load a previously trained model from disk.

        Returns:
            True if model loaded successfully.
        """
        if not os.path.exists(self.settings.trainer_file):
            return False

        try:
            self.model.read(self.settings.trainer_file)
            self._trained = True
            # Rebuild name cache
            with self.db.get_connection() as conn:
                for emp in self.db.get_all_employees(conn):
                    self._name_cache[emp["id"]] = emp["name"]
            logger.info("Model loaded from %s", self.settings.trainer_file)
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, face_roi: np.ndarray
                ) -> tuple[Optional[int], float, str]:
        """Predict the identity of a face region.

        Args:
            face_roi: Preprocessed grayscale face image (200x200).

        Returns:
            Tuple of (employee_id, confidence, name).
            ``employee_id`` is None if not recognized.
            Lower confidence = better match (LBPH convention).
        """
        if not self._trained:
            return None, 999.0, "unknown"

        try:
            uid, confidence = self.model.predict(face_roi)
        except Exception as e:
            logger.warning("Prediction error: %s", e)
            return None, 999.0, "unknown"

        threshold = self.settings.confidence_threshold
        if confidence < threshold and uid in self._name_cache:
            return uid, confidence, self._name_cache[uid]

        return None, confidence, "unknown"

    def reload_name_cache(self) -> None:
        """Rebuild the name-ID mapping from the database."""
        self._name_cache.clear()
        with self.db.get_connection() as conn:
            for emp in self.db.get_all_employees(conn):
                self._name_cache[emp["id"]] = emp["name"]
