"""
人脸识别打卡系统 - 主程序
实时摄像头人脸识别与打卡签到
"""
import os
import cv2
import numpy as np
from datetime import datetime
from utils import (
    TRAINER_FILE, LOGS_DIR, load_names, log_attendance, draw_ui
)

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 80  # 置信度阈值（越低表示匹配越好）


def run_attendance():
    """Run the real-time face recognition attendance system."""
    # Check model file
    if not os.path.exists(TRAINER_FILE):
        print("错误: 未找到训练模型，请先注册并训练模型。")
        print("请先运行: python face_register.py 和 python train_model.py")
        return

    names = load_names()
    if not names:
        print("错误: 没有已注册的员工。")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Use profile cascade for better side detection
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        return

    # For cooldown between repeated detections
    last_recognized = {}
    COOLDOWN_SECONDS = 5

    print(f"\n人脸识别打卡系统已启动 (共 {len(names)} 位员工)")
    print("=" * 40)
    print(f"{'姓名':<12} {'状态':<10} {'时间'}")
    print("-" * 40)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        # Also detect profile faces
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

        current_time = datetime.now()

        for (x, y, fw, fh) in faces:
            # Extract and preprocess face
            face_roi = gray[y:y + fh, x:x + fw]
            face_resized = cv2.resize(face_roi, (200, 200))

            # Equalize histogram for better recognition
            face_eq = cv2.equalizeHist(face_resized)

            # Predict
            try:
                uid, confidence = recognizer.predict(face_eq)
            except Exception:
                uid, confidence = -1, 999

            # Calculate confidence percentage (0 = perfect match)
            recognized = confidence < CONFIDENCE_THRESHOLD

            if recognized and uid in names:
                name = names[uid]
                status_text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)

                # Check cooldown
                cooldown_ok = (
                    uid not in last_recognized
                    or (current_time - last_recognized[uid]).total_seconds() > COOLDOWN_SECONDS
                )

                if cooldown_ok:
                    last_recognized[uid] = current_time
                    checked_in = log_attendance(name)
                    if checked_in:
                        timestamp = current_time.strftime("%H:%M:%S")
                        print(f"{name:<12} {'✓ 已打卡':<10} {timestamp}")
                        status_text = f"{name} ✓ 已打卡"
                    else:
                        status_text = f"{name} (已打卡)"
            else:
                status_text = "Unknown"
                color = (0, 0, 255)

            # Draw face rectangle
            cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 2)
            # Draw label background
            label_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.6, 2)[0]
            cv2.rectangle(display, (x, y - 30),
                          (x + label_size[0] + 10, y), color, -1)
            # Draw label text
            text_color = (0, 0, 0) if recognized else (255, 255, 255)
            cv2.putText(display, status_text, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Draw top bar
        now_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(display, f"人脸识别打卡系统 | {now_str}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw bottom hint
        cv2.rectangle(display, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "ESC:退出  R:查看今日记录  T:重新训练模型",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("人脸识别打卡系统", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            _show_today_records()
        elif key == ord('t') or key == ord('T'):
            cap.release()
            cv2.destroyAllWindows()
            print("\n正在重新训练模型...")
            from train_model import train_model
            train_model()
            recognizer.read(TRAINER_FILE)
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print("错误: 无法重新打开摄像头！")
                return
            print("模型已重新加载，继续打卡...")

    cap.release()
    cv2.destroyAllWindows()
    print("\n打卡系统已关闭。")


def _show_today_records():
    """Display today's attendance records."""
    log_path = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.csv")
    if not os.path.exists(log_path):
        print("\n今日暂无打卡记录。")
        return

    print(f"\n{'='*50}")
    print(f"  今日打卡记录 ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'='*50}")
    print(f"{'姓名':<12} {'时间':<10}")
    print("-" * 30)
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(",")
            if len(parts) >= 3:
                print(f"{parts[0]:<12} {parts[2]:<10}")
    print(f"{'='*50}\n")


def view_records():
    """View attendance records by date."""
    logs = sorted([f for f in os.listdir(LOGS_DIR) if f.endswith(".csv")],
                  reverse=True)
    if not logs:
        print("暂无打卡记录。")
        return

    print(f"\n{'='*50}")
    print("  打卡记录查询")
    print(f"{'='*50}")
    print(f"共有 {len(logs)} 天的记录:")
    print("-" * 50)

    for i, log in enumerate(logs[:10], 1):
        date = log.replace(".csv", "")
        with open(os.path.join(LOGS_DIR, log), "r", encoding="utf-8") as f:
            count = len(f.readlines()) - 1  # Exclude header
        print(f"  {i}. {date}  ({count}人打卡)")

    while True:
        try:
            choice = input("\n选择天数查看详情 (0返回): ").strip()
            if choice == "0":
                break
            idx = int(choice) - 1
            if 0 <= idx < min(len(logs), 10):
                date_str = logs[idx].replace(".csv", "")
                log_path = os.path.join(LOGS_DIR, logs[idx])
                print(f"\n{'='*50}")
                print(f"  {date_str} 打卡详情")
                print(f"{'='*50}")
                print(f"{'姓名':<12} {'时间':<10}")
                print("-" * 30)
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        parts = line.strip().split(",")
                        if len(parts) >= 3:
                            print(f"{parts[0]:<12} {parts[2]:<10}")
                print(f"{'='*50}\n")
                input("按 Enter 继续...")
                break
            else:
                print("无效选择")
        except ValueError:
            print("请输入数字")


if __name__ == "__main__":
    run_attendance()
