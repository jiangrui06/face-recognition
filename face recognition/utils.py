import os
import cv2
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance_logs")
TRAINER_FILE = os.path.join(DATA_DIR, "trainer.yml")
NAMES_FILE = os.path.join(DATA_DIR, "names.txt")

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def load_names():
    """Load name-id mapping from file."""
    names = {}
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        names[int(parts[0])] = parts[1]
    return names


def save_names(names):
    """Save name-id mapping to file."""
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        for uid, name in sorted(names.items()):
            f.write(f"{uid}:{name}\n")


def get_next_id(names):
    """Get next available user ID."""
    if not names:
        return 1
    return max(names.keys()) + 1


def get_today_log_path():
    """Get log file path for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOGS_DIR, f"{today}.csv")


def log_attendance(name):
    """Log attendance record with timestamp."""
    log_path = get_today_log_path()
    now = datetime.now().strftime("%H:%M:%S")
    date_today = datetime.now().strftime("%Y-%m-%d")

    header_needed = not os.path.exists(log_path)

    # Check if already checked in today (avoid duplicates)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{name},"):
                    return False  # Already checked in

    with open(log_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("姓名,日期,时间\n")
        f.write(f"{name},{date_today},{now}\n")
    return True


def draw_ui(frame, text, status_color=(0, 255, 0)):
    """Draw overlay UI on frame."""
    h, w = frame.shape[:2]
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    # Bottom hint
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Press ESC to exit | SPACE to confirm",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return frame
