"""Data models / dataclasses for the face recognition system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Employee:
    """Employee entity."""
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    sample_count: int = 0


@dataclass
class FaceSample:
    """Face image sample."""
    id: int
    employee_id: int
    image_path: str
    created_at: datetime


@dataclass
class AttendanceRecord:
    """Attendance check-in record."""
    id: int
    employee_id: int
    employee_name: str = ""
    check_in: datetime = field(default_factory=datetime.now)
    date: str = ""
    confidence: Optional[float] = None
    method: str = "lbph"


@dataclass
class RecognitionResult:
    """Face recognition result for a single frame."""
    employee_id: Optional[int]
    name: str
    confidence: float
    recognized: bool
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class AttendanceReport:
    """Attendance report for a date range."""
    start_date: str
    end_date: str
    total_employees: int
    total_records: int
    daily_records: dict[str, int]
    employee_records: list[dict]


@dataclass
class DashboardStats:
    """Dashboard statistics."""
    total_employees: int
    today_attendance: int
    today_rate: float
    monthly_attendance: int
    recent_records: list[AttendanceRecord]
