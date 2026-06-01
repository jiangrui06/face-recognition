"""摄像头管理模块 - 后台线程采集视频流，提供 MJPEG 编码帧。

摄像头不可用时不会阻塞应用，Web 界面仍可正常使用其他功能。
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.attendance import AttendanceService
from src.config import get_settings
from src.face_recognizer import FaceRecognizer
from src.utils import encode_frame_jpeg

logger = logging.getLogger(__name__)


class CameraManager:
    """后台线程管理摄像头采集和帧处理。

    线程安全地提供最新处理帧和识别事件。摄像头不可用时，
    ``available`` 返回 False，``get_frame()`` 返回空字节。
    """

    def __init__(self):
        self.settings = get_settings()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._available = False

        # 最新处理帧（JPEG 字节）
        self._frame: bytes = b""
        self._frame_available = False

        # 识别事件队列（最多 100 条）
        self._events: deque = deque(maxlen=100)

        # 跳帧计数器（性能优化）
        self._frame_count = 0
        self._process_every_n = 2

        # 业务服务
        self._recognizer = FaceRecognizer()
        self._recognizer.load_model()
        self._attendance = AttendanceService()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> bool:
        """启动摄像头后台采集线程。

        如果摄像头不可用，返回 False 但不抛出异常，
        Web 界面仍可正常使用。
        """
        if self._running:
            return True

        # 尝试打开摄像头
        self._cap = cv2.VideoCapture(self.settings.camera_index)
        if not self._cap or not self._cap.isOpened():
            logger.warning(
                "摄像头 %d 不可用，请检查："
                "1) 摄像头是否已连接  "
                "2) 是否有权限访问  "
                "3) config.yaml 中 camera.index 是否正确",
                self.settings.camera_index,
            )
            self._cap = None
            self._available = False
            return False

        # 设置分辨率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)

        self._running = True
        self._available = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True)
        self._thread.start()
        logger.info("摄像头已启动 (index=%d)", self.settings.camera_index)
        return True

    def stop(self) -> None:
        """停止摄像头采集。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._available = False
        logger.info("摄像头已停止")

    def restart(self) -> bool:
        """重启摄像头（例如重新训练模型后）。"""
        self.stop()
        time.sleep(0.5)
        return self.start()

    # ------------------------------------------------------------------
    # 帧获取
    # ------------------------------------------------------------------
    def get_frame(self) -> bytes:
        """获取最新处理帧（JPEG 字节）。"""
        with self._lock:
            return self._frame

    def has_frame(self) -> bool:
        """是否有可用帧。"""
        return self._frame_available

    def get_events(self) -> list[dict]:
        """获取并清空最近的识别事件。"""
        events = []
        with self._lock:
            while self._events:
                events.append(self._events.popleft())
        return events

    # ------------------------------------------------------------------
    # 内部采集循环
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """后台线程：持续采集和处理帧。"""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._frame_count += 1
            process = (self._frame_count % self._process_every_n == 0)

            if process and self._recognizer.is_trained:
                try:
                    annotated, events = self._attendance.process_frame(frame)
                    frame_to_encode = annotated
                    if events:
                        with self._lock:
                            for e in events:
                                self._events.append(e)
                except Exception as e:
                    logger.error("帧处理错误: %s", e)
                    frame_to_encode = frame
            else:
                frame_to_encode = frame

            # 编码为 JPEG
            jpeg = encode_frame_jpeg(frame_to_encode, quality=70)
            with self._lock:
                self._frame = jpeg
                self._frame_available = True

    def retrain_model(self) -> bool:
        """重新训练识别模型并重载。"""
        try:
            self._recognizer.train()
            self._recognizer.load_model()
            self._attendance.recognizer = self._recognizer
            logger.info("模型重训成功")
            return True
        except Exception as e:
            logger.error("模型重训失败: %s", e)
            return False
