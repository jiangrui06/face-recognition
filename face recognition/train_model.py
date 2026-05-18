"""
模型训练模块 - 使用LBPH算法训练人脸识别模型
"""
import os
import cv2
import numpy as np
from utils import FACES_DIR, TRAINER_FILE, load_names


def train_model():
    """Train LBPH face recognizer on collected face samples."""
    names = load_names()
    if not names:
        print("错误: 没有已注册的员工，请先注册人脸。")
        return False

    face_samples = []
    ids = []

    print("正在加载人脸样本...")
    for uid in names.keys():
        person_dir = os.path.join(FACES_DIR, str(uid))
        if not os.path.exists(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.endswith((".jpg", ".png")):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face_samples.append(img)
            ids.append(uid)

    if len(face_samples) < 10:
        print(f"错误: 样本不足 ({len(face_samples)}张)，至少需要10张。")
        return False

    print(f"共加载 {len(face_samples)} 个样本，{len(names)} 位员工。")

    # Train LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_FILE)

    print(f"✓ 模型训练完成，已保存至: {TRAINER_FILE}")
    return True


if __name__ == "__main__":
    train_model()
