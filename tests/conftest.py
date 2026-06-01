"""Test fixtures - temporary SQLite database for testing."""

import os
import tempfile
from pathlib import Path

import pytest

from src.database import DatabaseManager
from src.config import reload_settings


@pytest.fixture
def db_path():
    """Create a temporary database file path."""
    # Must close the handle for Windows compatibility
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    yield f.name
    if os.path.exists(f.name):
        try:
            os.unlink(f.name)
        except PermissionError:
            pass  # Windows may still lock it briefly


@pytest.fixture
def db_manager(db_path):
    """Create a DatabaseManager with a temporary database."""
    mgr = DatabaseManager(db_path)
    mgr.init_db()
    yield mgr
    # Close thread-local connection so next test gets a fresh one
    mgr.close()


@pytest.fixture
def populated_db(db_manager):
    """Database with sample employees and records."""
    with db_manager.get_connection() as conn:
        db_manager.add_employee(conn, "Alice")
        db_manager.add_employee(conn, "Bob")
        db_manager.add_employee(conn, "Charlie")

        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        conn.execute(
            "INSERT INTO attendance_records (employee_id, check_in, date) VALUES (?, ?, ?)",
            (1, f"{today}T09:00:00", today),
        )
        conn.execute(
            "INSERT INTO attendance_records (employee_id, check_in, date) VALUES (?, ?, ?)",
            (2, f"{today}T09:05:00", today),
        )
        conn.execute(
            "INSERT INTO attendance_records (employee_id, check_in, date) VALUES (?, ?, ?)",
            (1, f"{yesterday}T09:00:00", yesterday),
        )
    return db_manager


@pytest.fixture
def reset_settings():
    """Reset settings cache before and after tests."""
    reload_settings()
    yield
    reload_settings()
