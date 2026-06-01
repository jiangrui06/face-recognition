"""Face registration module - capture and save face samples.

Provides both CLI and API-friendly registration flows.
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings
from src.database import DatabaseManager
from src.exceptions import RegistrationError
from src.face_detector import create_detector, FaceDetector
from src.utils import is_blurry

logger = logging.getLogger(__name__)


class FaceRegister:
    """Handles employee face registration and sample collection."""

    def __init__(self):
        self.settings = get_settings()
        self.db = DatabaseManager()
        self.detector: FaceDetector = create_detector(
            self.settings.detector_type
        )

    def register(self, name: str, cap: Optional[cv2.VideoCapture] = None
                 ) -> dict:
        """Register a new employee by capturing face samples.

        Args:
            name: Employee name.
            cap: OpenCV VideoCapture (created internally if None).

        Returns:
            dict with keys: success, employee_id, sample_count, message.
        """
        name = name.strip()
        if not name:
            raise RegistrationError("姓名不能为空")

        with self.db.get_connection() as conn:
            existing = self.db.get_employee_by_name(conn, name)
            if existing:
                raise RegistrationError(
                    f"员工 '{name}' 已存在 (ID: {existing['id']})"
                )
            emp_id = self.db.add_employee(conn, name)

        faces_dir = self.settings.faces_dir
        person_dir = os.path.join(faces_dir, str(emp_id))
        os.makedirs(person_dir, exist_ok=True)

        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(self.settings.camera_index)
            own_cap = True
            if not cap.isOpened():
                raise RegistrationError("Cannot open camera")

        try:
            count = 0
            max_samples = self.settings.face_sample_count
            quality_count = 0

            while count < max_samples:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detect(gray, min_size=100)

                display = frame.copy()
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(display, (x, y),
                                  (x + fw, y + fh), (0, 255, 0), 2)

                    face_roi = gray[y:y + fh, x:x + fw]

                    # Skip blurry samples
                    if is_blurry(face_roi):
                        label = "Blurry, move closer"
                        cv2.putText(display, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 1)
                        continue

                    count += 1
                    face_resized = cv2.resize(face_roi, (200, 200))
                    img_path = os.path.join(person_dir, f"{count}.jpg")
                    cv2.imwrite(img_path, face_resized)

                    with self.db.get_connection() as conn:
                        self.db.add_face_sample(conn, emp_id, img_path)

                    quality_count += 1
                    cv2.putText(display, f"Captured: {count}/{max_samples}",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                # Status bar
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(display,
                            f"Registering: {name} | Progress: {count}/{max_samples}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
                cv2.putText(display,
                            "Slowly turn your head | ESC to cancel",
                            (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1)

                cv2.imshow("Face Registration", display)
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    logger.info("Registration cancelled for '%s'", name)
                    break

            success = count >= max_samples * 0.5
            if not success:
                # Clean up samples if not enough
                for f in os.listdir(person_dir):
                    os.remove(os.path.join(person_dir, f))
                os.rmdir(person_dir)
                with self.db.get_connection() as conn:
                    self.db.delete_employee(conn, emp_id)

            logger.info("Registration '%s': %d/%d samples (success=%s)",
                        name, count, max_samples, success)
            return {
                "success": success,
                "employee_id": emp_id,
                "sample_count": count,
                "message": (f"员工 '{name}' 注册成功，共采集 {count} 张样本" if success
                            else "注册失败，样本不足"),
            }

        finally:
            if own_cap:
                cap.release()
            cv2.destroyAllWindows()

    def list_employees(self) -> list[dict]:
        """List all employees with sample counts."""
        with self.db.get_connection() as conn:
            return self.db.get_employees_with_sample_count(conn)
