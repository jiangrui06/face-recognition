"""Tests for the database layer."""

import pytest


class TestDatabaseInit:
    """Test database initialization."""

    def test_init_creates_tables(self, db_manager):
        """Verify that init_db() creates all required tables."""
        with db_manager.get_connection() as conn:
            tables = db_manager.fetch_all(
                conn,
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            )
            table_names = [t["name"] for t in tables]
            assert "employees" in table_names
            assert "face_samples" in table_names
            assert "attendance_records" in table_names

    def test_init_is_idempotent(self, db_manager):
        """Calling init_db() twice should not raise."""
        db_manager.init_db()  # second call
        with db_manager.get_connection() as conn:
            count = db_manager.fetch_one(
                conn,
                "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table'",
            )
            assert count["cnt"] >= 3


class TestEmployeeCRUD:
    """Test employee CRUD operations."""

    def test_add_employee(self, db_manager):
        with db_manager.get_connection() as conn:
            emp_id = db_manager.add_employee(conn, "Alice")
            assert emp_id > 0

    def test_add_duplicate_name_raises(self, db_manager):
        with db_manager.get_connection() as conn:
            db_manager.add_employee(conn, "Alice")
            with pytest.raises(Exception):
                db_manager.add_employee(conn, "Alice")

    def test_get_employee_by_name(self, db_manager):
        with db_manager.get_connection() as conn:
            db_manager.add_employee(conn, "Alice")
            emp = db_manager.get_employee_by_name(conn, "Alice")
            assert emp is not None
            assert emp["name"] == "Alice"

    def test_get_employee_by_name_not_found(self, db_manager):
        with db_manager.get_connection() as conn:
            emp = db_manager.get_employee_by_name(conn, "Nonexistent")
            assert emp is None

    def test_get_employee_by_id(self, db_manager):
        with db_manager.get_connection() as conn:
            emp_id = db_manager.add_employee(conn, "Bob")
            emp = db_manager.get_employee_by_id(conn, emp_id)
            assert emp is not None
            assert emp["name"] == "Bob"

    def test_get_all_employees(self, db_manager):
        with db_manager.get_connection() as conn:
            db_manager.add_employee(conn, "A")
            db_manager.add_employee(conn, "B")
            db_manager.add_employee(conn, "C")
            emps = db_manager.get_all_employees(conn)
            assert len(emps) == 3

    def test_get_all_employees_active_only(self, db_manager):
        with db_manager.get_connection() as conn:
            db_manager.add_employee(conn, "A")
            db_manager.add_employee(conn, "B")
            db_manager.delete_employee(conn, 2)
            emps = db_manager.get_all_employees(conn, active_only=True)
            assert len(emps) == 1

    def test_delete_employee_soft(self, db_manager):
        with db_manager.get_connection() as conn:
            emp_id = db_manager.add_employee(conn, "ToDelete")
            assert db_manager.delete_employee(conn, emp_id) is True
            emp = db_manager.get_employee_by_id(conn, emp_id)
            assert emp["is_active"] == 0

    def test_delete_nonexistent_employee(self, db_manager):
        with db_manager.get_connection() as conn:
            assert db_manager.delete_employee(conn, 999) is False

    def test_update_employee_name(self, db_manager):
        with db_manager.get_connection() as conn:
            emp_id = db_manager.add_employee(conn, "OldName")
            assert db_manager.update_employee_name(conn, emp_id, "NewName")
            emp = db_manager.get_employee_by_id(conn, emp_id)
            assert emp["name"] == "NewName"


class TestAttendanceRecords:
    """Test attendance record operations."""

    def test_add_attendance(self, db_manager):
        with db_manager.get_connection() as conn:
            emp_id = db_manager.add_employee(conn, "Alice")
            rec_id = db_manager.add_attendance(conn, emp_id, 75.0)
            assert rec_id > 0

    def test_has_checked_in_today(self, populated_db):
        with populated_db.get_connection() as conn:
            assert populated_db.has_checked_in_today(conn, 1) is True
            assert populated_db.has_checked_in_today(conn, 3) is False

    def test_get_today_attendance(self, populated_db):
        records = populated_db.get_today_attendance(
            populated_db.get_connection().__enter__()
        )[:0]  # Hmm, need to fix the API usage
        # Actually let's do it properly
        with populated_db.get_connection() as conn:
            records = populated_db.get_today_attendance(conn)
            assert len(records) == 2
            names = {r["employee_name"] for r in records}
            assert "Alice" in names
            assert "Bob" in names

    def test_get_today_count(self, populated_db):
        with populated_db.get_connection() as conn:
            count = populated_db.get_today_count(conn)
            assert count == 2

    def test_get_daily_counts(self, populated_db):
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        with populated_db.get_connection() as conn:
            counts = populated_db.get_daily_counts(conn, yesterday, today)
            count_dict = {c["date"]: c["count"] for c in counts}
            assert count_dict.get(today) == 2
            assert count_dict.get(yesterday) == 1

    def test_get_attendance_report(self, populated_db):
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        with populated_db.get_connection() as conn:
            records = populated_db.get_attendance_report(conn, yesterday, today)
            assert len(records) >= 3


class TestEmployeesWithSampleCount:
    """Test employee queries with sample counts."""

    def test_get_employees_with_sample_count(self, db_manager):
        with db_manager.get_connection() as conn:
            db_manager.add_employee(conn, "Alice")
            results = db_manager.get_employees_with_sample_count(conn)
            assert len(results) == 1
            assert results[0]["sample_count"] == 0


class TestMigration:
    """Test legacy data migration."""

    def test_migrate_from_legacy_no_files(self, db_manager, tmp_path):
        result = db_manager.migrate_from_legacy(
            str(tmp_path / "nonexistent.txt"),
            str(tmp_path / "nonexistent_dir"),
        )
        assert result["employees"] == 0
        assert result["records"] == 0
