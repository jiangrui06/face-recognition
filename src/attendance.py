"""Attendance service - check-in, records, reports, and real-time recognition."""

import csv
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np

from src.config import get_settings
from src.database import DatabaseManager
from src.face_detector import create_detector, FaceDetector
from src.face_recognizer import FaceRecognizer
from src.models import AttendanceRecord, AttendanceReport, DashboardStats
from src.utils import draw_detection, preprocess_face, encode_frame_jpeg

logger = logging.getLogger(__name__)


class AttendanceService:
    """Core attendance business logic."""

    def __init__(self):
        self.settings = get_settings()
        self.db = DatabaseManager()
        self.recognizer = FaceRecognizer()
        self.detector: FaceDetector = create_detector(
            self.settings.detector_type
        )
        self._last_checkin: dict[int, datetime] = {}

    # ------------------------------------------------------------------
    # Check-in
    # ------------------------------------------------------------------
    def check_in(self, employee_id: int, confidence: float = 0.0,
                 method: str = "lbph") -> bool:
        """Record attendance for an employee.

        Returns False if already checked in today (no duplicate).
        """
        with self.db.get_connection() as conn:
            if self.db.has_checked_in_today(conn, employee_id):
                return False
            self.db.add_attendance(conn, employee_id, confidence, method)
            return True

    def process_frame(self, frame: np.ndarray
                      ) -> tuple[np.ndarray, list[dict]]:
        """Process a single frame: detect faces, recognize, draw results.

        Returns:
            Tuple of (annotated_frame, recognition_events).
            Each event dict: {employee_id, name, confidence, checked_in}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(gray, self.settings.face_min_size)
        now = datetime.now()
        events = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            processed = preprocess_face(face_roi)
            uid, confidence, name = self.recognizer.predict(processed)

            recognized = uid is not None
            cooldown_ok = self._check_cooldown(uid, now)

            if recognized and cooldown_ok:
                self._last_checkin[uid] = now
                checked_in = self.check_in(uid, confidence)

                label = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)

                if checked_in:
                    logger.info("Check-in: %s (conf=%.1f)", name, confidence)
                    events.append({
                        "employee_id": uid,
                        "name": name,
                        "confidence": confidence,
                        "checked_in": True,
                    })
            else:
                label = f"Unknown ({confidence:.1f})" if not recognized else name
                color = (0, 0, 255) if not recognized else (0, 255, 0)

            draw_detection(frame, x, y, w, h, label, color)

        return frame, events

    def _check_cooldown(self, employee_id: Optional[int],
                        now: datetime) -> bool:
        if employee_id is None:
            return False
        last = self._last_checkin.get(employee_id)
        if last is None:
            return True
        return (now - last).total_seconds() > self.settings.cooldown_seconds

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_today_records(self) -> list[AttendanceRecord]:
        with self.db.get_connection() as conn:
            rows = self.db.get_today_attendance(conn)
        return [self._row_to_record(r) for r in rows]

    def get_records_by_date(self, date: str) -> list[AttendanceRecord]:
        with self.db.get_connection() as conn:
            rows = self.db.get_attendance_by_date(conn, date)
        return [self._row_to_record(r) for r in rows]

    def get_report(self, start_date: str, end_date: str) -> AttendanceReport:
        with self.db.get_connection() as conn:
            employees = self.db.get_all_employees(conn)
            records = self.db.get_attendance_report(conn, start_date, end_date)
            daily_counts = self.db.get_daily_counts(conn, start_date, end_date)

        total_employees = len(employees)
        total_records = len(records)
        daily_records = {d["date"]: d["count"] for d in daily_counts}

        # Per-employee summary
        emp_summary: dict[str, int] = {}
        for r in records:
            name = r["employee_name"]
            emp_summary[name] = emp_summary.get(name, 0) + 1

        employee_records = [
            {"name": name, "days": count}
            for name, count in sorted(emp_summary.items(),
                                       key=lambda x: -x[1])
        ]

        return AttendanceReport(
            start_date=start_date,
            end_date=end_date,
            total_employees=total_employees,
            total_records=total_records,
            daily_records=daily_records,
            employee_records=employee_records,
        )

    def get_dashboard_stats(self) -> DashboardStats:
        with self.db.get_connection() as conn:
            total = len(self.db.get_all_employees(conn))
            today_count = self.db.get_today_count(conn)
            monthly_count = self.db.get_monthly_count(conn)
            recent = self.db.get_today_attendance(conn)

        rate = (today_count / total * 100) if total > 0 else 0.0
        records = [self._row_to_record(r) for r in recent]

        return DashboardStats(
            total_employees=total,
            today_attendance=today_count,
            today_rate=round(rate, 1),
            monthly_attendance=monthly_count,
            recent_records=records[:10],
        )

    def export_csv(self, date: str, output_path: str) -> str:
        """Export attendance records for a date to CSV."""
        records = self.get_records_by_date(date)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time", "Confidence", "Method"])
            for r in records:
                time_str = r.check_in.strftime("%H:%M:%S") if isinstance(
                    r.check_in, datetime) else str(r.check_in)[11:19]
                writer.writerow([
                    r.employee_name, r.date, time_str,
                    f"{r.confidence:.1f}" if r.confidence else "",
                    r.method,
                ])
        return output_path

    @staticmethod
    def _row_to_record(row: dict) -> AttendanceRecord:
        check_in = row.get("check_in", "")
        if isinstance(check_in, str):
            try:
                check_in = datetime.fromisoformat(check_in)
            except ValueError:
                check_in = datetime.now()
        return AttendanceRecord(
            id=row["id"],
            employee_id=row["employee_id"],
            employee_name=row.get("employee_name", ""),
            check_in=check_in,
            date=row.get("date", ""),
            confidence=row.get("confidence"),
            method=row.get("method", "lbph"),
        )
