"""
人脸注册模块 - 采集人脸图像并保存
"""
import os
import cv2
from utils import FACES_DIR, load_names, save_names, get_next_id

FACE_SAMPLES = 60  # 每人采集样本数
CAMERA_INDEX = 0


def register_face():
    """Register a new face through camera capture."""
    names = load_names()
    name = input("请输入员工姓名: ").strip()
    if not name:
        print("姓名不能为空！")
        return

    # Check if name already exists
    for uid, n in names.items():
        if n == name:
            print(f"员工 '{name}' 已存在 (ID: {uid})")
            return

    uid = get_next_id(names)
    person_dir = os.path.join(FACES_DIR, str(uid))
    os.makedirs(person_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        return

    count = 0
    print(f"\n正在为 '{name}' 采集人脸样本...")
    print("请正对摄像头，保持面部在绿色方框中。")

    while count < FACE_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
            # Save face sample
            face_img = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_img, (200, 200))
            path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(path, face_resized)
            cv2.putText(frame, f"采集: {count}/{FACE_SAMPLES}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Info display
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"正在注册: {name} | 采集进度: {count}/{FACE_SAMPLES}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "请缓慢转动头部以采集多角度样本",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("人脸注册", frame)
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            print("采集已取消")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= FACE_SAMPLES * 0.5:  # At least 50% collected
        names[uid] = name
        save_names(names)
        print(f"\n✓ 员工 '{name}' 注册成功！共采集 {count} 张人脸样本。")
    else:
        print(f"\n✗ 采集失败，样本不足 ({count}/{FACE_SAMPLES})")


def list_registered():
    """List all registered users."""
    names = load_names()
    if not names:
        print("当前没有已注册的员工。")
        return
    print(f"\n已注册员工 ({len(names)}人):")
    print("-" * 30)
    for uid in sorted(names.keys()):
        person_dir = os.path.join(FACES_DIR, str(uid))
        sample_count = len([f for f in os.listdir(person_dir)
                           if f.endswith(".jpg")]) if os.path.exists(person_dir) else 0
        print(f"  ID {uid:3d} | {names[uid]:10s} | {sample_count} 个样本")


if __name__ == "__main__":
    register_face()
