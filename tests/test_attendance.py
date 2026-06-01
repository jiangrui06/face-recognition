"""Tests for the attendance service."""

from datetime import datetime, timedelta

import pytest

from src.attendance import AttendanceService
from src.database import DatabaseManager
from src.models import AttendanceReport


@pytest.fixture
def service_with_data(db_path):
    """Create AttendanceService with a populated database."""
    db = DatabaseManager(db_path)
    db.init_db()

    with db.get_connection() as conn:
        db.add_employee(conn, "Alice")
        db.add_employee(conn, "Bob")
        db.add_employee(conn, "Charlie")

    svc = AttendanceService()
    # Override the db so it uses our test db
    svc.db = db
    return svc


class TestCheckIn:
    """Test check-in logic."""

    def test_check_in_success(self, service_with_data):
        result = service_with_data.check_in(1, 75.0)
        assert result is True

    def test_check_in_no_duplicate(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        result = service_with_data.check_in(1, 70.0)
        assert result is False

    def test_check_in_multiple_employees(self, service_with_data):
        assert service_with_data.check_in(1, 75.0) is True
        assert service_with_data.check_in(2, 80.0) is True
        assert service_with_data.check_in(3, 65.0) is True

    def test_check_in_returns_false_for_duplicate(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        assert service_with_data.check_in(1, 70.0) is False


class TestRecords:
    """Test record retrieval."""

    def test_get_today_records_empty(self, service_with_data):
        records = service_with_data.get_today_records()
        assert len(records) == 0

    def test_get_today_records_after_checkin(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        records = service_with_data.get_today_records()
        assert len(records) == 1
        assert records[0].employee_name == "Alice"

    def test_get_records_by_date(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        today = datetime.now().strftime("%Y-%m-%d")
        records = service_with_data.get_records_by_date(today)
        assert len(records) == 1


class TestReport:
    """Test report generation."""

    def test_get_report_empty_range(self, service_with_data):
        report = service_with_data.get_report("2099-01-01", "2099-12-31")
        assert report.total_records == 0
        assert report.total_employees == 3

    def test_get_report_with_data(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        service_with_data.check_in(2, 80.0)
        today = datetime.now().strftime("%Y-%m-%d")
        report = service_with_data.get_report(today, today)
        assert report.total_records == 2
        assert report.total_employees == 3

    def test_report_type(self, service_with_data):
        today = datetime.now().strftime("%Y-%m-%d")
        report = service_with_data.get_report(today, today)
        assert isinstance(report, AttendanceReport)


class TestDashboardStats:
    """Test dashboard statistics."""

    def test_get_dashboard_stats_empty(self, service_with_data):
        stats = service_with_data.get_dashboard_stats()
        assert stats.total_employees == 3
        assert stats.today_attendance == 0
        assert stats.today_rate == 0.0

    def test_get_dashboard_stats_after_checkin(self, service_with_data):
        service_with_data.check_in(1, 75.0)
        stats = service_with_data.get_dashboard_stats()
        assert stats.today_attendance == 1
        assert stats.today_rate == pytest.approx(33.3, rel=0.1)


class TestExport:
    """Test CSV export."""

    def test_export_csv_empty(self, service_with_data, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        output = tmp_path / "test.csv"
        result = service_with_data.export_csv(today, str(output))
        assert result == str(output)

    def test_export_csv_with_data(self, service_with_data, tmp_path):
        service_with_data.check_in(1, 75.0)
        today = datetime.now().strftime("%Y-%m-%d")
        output = tmp_path / "test.csv"
        result = service_with_data.export_csv(today, str(output))
        with open(result, "r", encoding="utf-8-sig") as f:
            content = f.read()
            assert "Alice" in content
            assert "75" in content
