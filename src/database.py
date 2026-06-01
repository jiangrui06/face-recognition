"""SQLite database layer - connection management, schema migration, CRUD."""

import csv
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import get_settings
from src.exceptions import DatabaseError

logger = logging.getLogger(__name__)

_local = threading.local()


class DatabaseManager:
    """Thread-safe SQLite database manager.

    Caches one connection per thread per database path.
    Usage::

        db = DatabaseManager()
        db.init_db()

        with db.get_connection() as conn:
            rows = db.fetch_all(conn, "SELECT * FROM employees")
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or get_settings().db_path
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection (context manager).

        Connections are cached per thread per database path.
        """
        if not hasattr(_local, "conns"):
            _local.conns = {}

        if self._db_path not in _local.conns or _local.conns[self._db_path] is None:
            _local.conns[self._db_path] = self._create_connection()

        conn = _local.conns[self._db_path]
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        """Close the connection for this instance's database path."""
        if hasattr(_local, "conns") and self._db_path in _local.conns:
            conn = _local.conns.pop(self._db_path, None)
            if conn:
                conn.close()

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------
    def init_db(self) -> None:
        """Create tables and indexes if they don't exist (idempotent)."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS attendance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER NOT NULL REFERENCES employees(id),
                    check_in TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    date TEXT NOT NULL,
                    confidence REAL,
                    method TEXT DEFAULT 'lbph'
                );

                CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance_records(date);
                CREATE INDEX IF NOT EXISTS idx_attendance_employee_date
                    ON attendance_records(employee_id, date);
                CREATE INDEX IF NOT EXISTS idx_face_samples_employee
                    ON face_samples(employee_id);
            """)
            logger.info("Database initialized: %s", self._db_path)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def fetch_one(self, conn: sqlite3.Connection, sql: str,
                  params: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch a single row."""
        return conn.execute(sql, params).fetchone()

    def fetch_all(self, conn: sqlite3.Connection, sql: str,
                  params: tuple = ()) -> list[sqlite3.Row]:
        """Fetch all matching rows."""
        return conn.execute(sql, params).fetchall()

    def execute(self, conn: sqlite3.Connection, sql: str,
                params: tuple = ()) -> sqlite3.Cursor:
        """Execute a write operation."""
        return conn.execute(sql, params)

    def execute_many(self, conn: sqlite3.Connection, sql: str,
                     seq: list[tuple]) -> None:
        """Execute many write operations."""
        conn.executemany(sql, seq)

    # ------------------------------------------------------------------
    # Employee CRUD
    # ------------------------------------------------------------------
    def add_employee(self, conn: sqlite3.Connection, name: str) -> int:
        """Add a new employee. Returns the new ID."""
        cursor = conn.execute(
            "INSERT INTO employees (name) VALUES (?)", (name,)
        )
        logger.info("Employee added: %s (id=%d)", name, cursor.lastrowid)
        return cursor.lastrowid

    def get_employee_by_name(self, conn: sqlite3.Connection,
                             name: str) -> Optional[sqlite3.Row]:
        return self.fetch_one(conn, "SELECT * FROM employees WHERE name = ?",
                              (name,))

    def get_employee_by_id(self, conn: sqlite3.Connection,
                           employee_id: int) -> Optional[sqlite3.Row]:
        return self.fetch_one(conn, "SELECT * FROM employees WHERE id = ?",
                              (employee_id,))

    def get_all_employees(self, conn: sqlite3.Connection,
                          active_only: bool = True) -> list[sqlite3.Row]:
        if active_only:
            return self.fetch_all(
                conn, "SELECT * FROM employees WHERE is_active = 1 ORDER BY id"
            )
        return self.fetch_all(conn, "SELECT * FROM employees ORDER BY id")

    def delete_employee(self, conn: sqlite3.Connection,
                        employee_id: int) -> bool:
        cursor = conn.execute(
            "UPDATE employees SET is_active = 0 WHERE id = ?",
            (employee_id,)
        )
        return cursor.rowcount > 0

    def update_employee_name(self, conn: sqlite3.Connection,
                             employee_id: int, new_name: str) -> bool:
        cursor = conn.execute(
            "UPDATE employees SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_name, employee_id),
        )
        return cursor.rowcount > 0

    def get_employee_sample_count(self, conn: sqlite3.Connection,
                                  employee_id: int) -> int:
        row = self.fetch_one(
            conn, "SELECT COUNT(*) as cnt FROM face_samples WHERE employee_id = ?",
            (employee_id,)
        )
        return row["cnt"] if row else 0

    def get_employees_with_sample_count(
            self, conn: sqlite3.Connection
    ) -> list[dict]:
        """Get all employees with their sample counts."""
        rows = self.fetch_all(conn, """
            SELECT e.*, COUNT(fs.id) as sample_count
            FROM employees e
            LEFT JOIN face_samples fs ON fs.employee_id = e.id AND fs.employee_id IS NOT NULL
            WHERE e.is_active = 1
            GROUP BY e.id
            ORDER BY e.id
        """)
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Face samples
    # ------------------------------------------------------------------
    def add_face_sample(self, conn: sqlite3.Connection,
                        employee_id: int, image_path: str) -> int:
        cursor = conn.execute(
            "INSERT INTO face_samples (employee_id, image_path) VALUES (?, ?)",
            (employee_id, image_path)
        )
        return cursor.lastrowid

    def get_face_samples(self, conn: sqlite3.Connection,
                         employee_id: int) -> list[sqlite3.Row]:
        return self.fetch_all(
            conn, "SELECT * FROM face_samples WHERE employee_id = ? ORDER BY id",
            (employee_id,)
        )

    def delete_face_samples(self, conn: sqlite3.Connection,
                            employee_id: int) -> None:
        self.execute(conn, "DELETE FROM face_samples WHERE employee_id = ?",
                     (employee_id,))

    # ------------------------------------------------------------------
    # Attendance records
    # ------------------------------------------------------------------
    def add_attendance(self, conn: sqlite3.Connection,
                       employee_id: int, confidence: float = 0.0,
                       method: str = "lbph") -> int:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        cursor = conn.execute(
            """INSERT INTO attendance_records
               (employee_id, check_in, date, confidence, method)
               VALUES (?, ?, ?, ?, ?)""",
            (employee_id, now.isoformat(), date_str, confidence, method)
        )
        return cursor.lastrowid

    def has_checked_in_today(self, conn: sqlite3.Connection,
                             employee_id: int) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.fetch_one(
            conn,
            "SELECT id FROM attendance_records WHERE employee_id = ? AND date = ? LIMIT 1",
            (employee_id, today),
        )
        return row is not None

    def get_today_attendance(self, conn: sqlite3.Connection
                             ) -> list[dict]:
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self.fetch_all(conn, """
            SELECT a.*, e.name as employee_name
            FROM attendance_records a
            JOIN employees e ON e.id = a.employee_id
            WHERE a.date = ?
            ORDER BY a.check_in DESC
        """, (today,))
        return [dict(r) for r in rows]

    def get_attendance_by_date(self, conn: sqlite3.Connection,
                               date: str) -> list[dict]:
        rows = self.fetch_all(conn, """
            SELECT a.*, e.name as employee_name
            FROM attendance_records a
            JOIN employees e ON e.id = a.employee_id
            WHERE a.date = ?
            ORDER BY a.check_in DESC
        """, (date,))
        return [dict(r) for r in rows]

    def get_attendance_report(self, conn: sqlite3.Connection,
                              start_date: str, end_date: str) -> list[dict]:
        rows = self.fetch_all(conn, """
            SELECT a.date, e.name as employee_name, a.check_in, a.confidence
            FROM attendance_records a
            JOIN employees e ON e.id = a.employee_id
            WHERE a.date >= ? AND a.date <= ?
            ORDER BY a.date DESC, a.check_in DESC
        """, (start_date, end_date))
        return [dict(r) for r in rows]

    def get_daily_counts(self, conn: sqlite3.Connection,
                         start_date: str, end_date: str) -> list[dict]:
        rows = self.fetch_all(conn, """
            SELECT date, COUNT(DISTINCT employee_id) as count
            FROM attendance_records
            WHERE date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date
        """, (start_date, end_date))
        return [dict(r) for r in rows]

    def get_today_count(self, conn: sqlite3.Connection) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.fetch_one(
            conn, "SELECT COUNT(DISTINCT employee_id) as cnt FROM attendance_records WHERE date = ?",
            (today,)
        )
        return row["cnt"] if row else 0

    def get_monthly_count(self, conn: sqlite3.Connection) -> int:
        month = datetime.now().strftime("%Y-%m")
        row = self.fetch_one(
            conn,
            "SELECT COUNT(DISTINCT employee_id) as cnt FROM attendance_records WHERE date LIKE ?",
            (f"{month}%",)
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Data migration from old CSV format
    # ------------------------------------------------------------------
    def migrate_from_legacy(self, names_file: str,
                            logs_dir: str) -> dict[str, int]:
        """Migrate from old names.txt and CSV logs to SQLite.

        Returns counts: {"employees": N, "records": N}
        """
        result = {"employees": 0, "records": 0}

        with self.get_connection() as conn:
            # Migrate names.txt -> employees
            if os.path.exists(names_file):
                name_map = {}  # old_id -> new_id
                with open(names_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if ":" not in line:
                            continue
                        old_id, name = line.split(":", 1)
                        existing = self.get_employee_by_name(conn, name)
                        if existing:
                            name_map[int(old_id)] = existing["id"]
                        else:
                            new_id = self.add_employee(conn, name)
                            name_map[int(old_id)] = new_id
                            result["employees"] += 1

            # Migrate CSV logs -> attendance_records
            if os.path.exists(logs_dir):
                for csv_file in sorted(os.listdir(logs_dir)):
                    if not csv_file.endswith(".csv"):
                        continue
                    date_str = csv_file.replace(".csv", "")
                    csv_path = os.path.join(logs_dir, csv_file)
                    with open(csv_path, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        next(reader, None)  # skip header
                        for row in reader:
                            if len(row) < 3:
                                continue
                            emp_name, _, time_str = row[0], row[1], row[2]
                            emp_row = self.get_employee_by_name(conn, emp_name)
                            if not emp_row:
                                continue
                            check_in = f"{date_str}T{time_str}"
                            conn.execute(
                                """INSERT OR IGNORE INTO attendance_records
                                   (employee_id, check_in, date, method)
                                   VALUES (?, ?, ?, 'legacy')""",
                                (emp_row["id"], check_in, date_str),
                            )
                            result["records"] += 1

        logger.info("Migration complete: %d employees, %d records",
                    result["employees"], result["records"])
        return result
