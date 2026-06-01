"""FastAPI Web 应用 - REST API 和页面路由。

提供：
- 仪表盘 - 考勤统计概览
- 实时人脸识别 - MJPEG 视频流
- 员工管理 - 增删查改
- 统计报表 - 查询和导出
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.attendance import AttendanceService
from src.config import get_settings
from src.database import DatabaseManager
from src.face_recognizer import FaceRecognizer
from src.face_register import FaceRegister
from src.exceptions import RegistrationError
from src.models import DashboardStats
from src.utils import encode_frame_jpeg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
settings = get_settings()
project_root = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="人脸识别打卡系统",
    version="2.0.0",
    description="全栈人脸识别打卡系统 REST API - 支持实时识别、考勤管理、统计报表",
)

# Mount static files
static_dir = project_root / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = project_root / "web" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Services
db = DatabaseManager()
attendance_service = AttendanceService()
register_service = FaceRegister()
recognizer = FaceRecognizer()
recognizer.load_model()

# Camera manager (lazy init on first use)
_camera = None


def get_camera():
    global _camera
    if _camera is None:
        from web.camera import CameraManager
        _camera = CameraManager()
    return _camera


# Ensure DB initialized
db.init_db()


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def _register_template_filters():
    """Register custom Jinja2 filters."""
    templates.env.filters["datetime"] = lambda v: (
        v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, "strftime") else str(v)[:19]
    )


_register_template_filters()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """仪表盘页面 - 显示系统概览和考勤统计。"""
    stats = attendance_service.get_dashboard_stats()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stats": stats},
    )


@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page(request: Request):
    """实时打卡页面 - 视频流人脸识别。"""
    return templates.TemplateResponse(
        "attendance.html",
        {"request": request},
    )


@app.get("/employees", response_class=HTMLResponse)
async def employees_page(request: Request):
    """员工管理页面 - 注册、查看和管理员工。"""
    employees_data = register_service.list_employees()
    return templates.TemplateResponse(
        "employees.html",
        {"request": request, "employees": employees_data},
    )


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """统计报表页面 - 查询和导出考勤数据。"""
    stats = attendance_service.get_dashboard_stats()
    return templates.TemplateResponse(
        "reports.html",
        {"request": request, "stats": stats},
    )


# ---------------------------------------------------------------------------
# API - 员工管理
# ---------------------------------------------------------------------------
@app.get("/api/employees")
async def api_list_employees():
    """获取所有活跃员工列表（含样本数）。"""
    return register_service.list_employees()


@app.post("/api/employees")
async def api_register_employee(name: str = Form(...)):
    """注册新员工。"""
    try:
        result = register_service.register(name)
        return {"success": True, "data": result}
    except RegistrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Employee registration failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/employees/{employee_id}")
async def api_delete_employee(employee_id: int):
    """删除员工（软删除，设置 is_active=0）。"""
    with db.get_connection() as conn:
        success = db.delete_employee(conn, employee_id)
    if not success:
        raise HTTPException(status_code=404, detail="员工不存在")
    return {"success": True}


# ---------------------------------------------------------------------------
# API - Attendance
# ---------------------------------------------------------------------------
@app.get("/api/attendance/today")
async def api_today_attendance():
    """获取今日打卡记录。"""
    records = attendance_service.get_today_records()
    return {
        "success": True,
        "count": len(records),
        "records": [
            {
                "id": r.id,
                "employee_id": r.employee_id,
                "name": r.employee_name,
                "time": r.check_in.strftime("%H:%M:%S"),
                "confidence": r.confidence,
                "method": r.method,
            }
            for r in records
        ],
    }


@app.get("/api/attendance/date")
async def api_attendance_by_date(date: str = Query(..., description="YYYY-MM-DD")):
    """获取指定日期的打卡记录。"""
    records = attendance_service.get_records_by_date(date)
    return {
        "success": True,
        "date": date,
        "count": len(records),
        "records": [
            {
                "id": r.id,
                "employee_id": r.employee_id,
                "name": r.employee_name,
                "time": r.check_in.strftime("%H:%M:%S"),
                "confidence": r.confidence,
            }
            for r in records
        ],
    }


@app.get("/api/attendance/report")
async def api_report(start_date: str = Query(...),
                     end_date: str = Query(...)):
    """获取指定日期范围的考勤报表。"""
    report = attendance_service.get_report(start_date, end_date)
    return {
        "success": True,
        "report": {
            "start_date": report.start_date,
            "end_date": report.end_date,
            "total_employees": report.total_employees,
            "total_records": report.total_records,
            "daily_records": report.daily_records,
            "employee_records": report.employee_records,
        },
    }


@app.get("/api/attendance/export")
async def api_export_csv(date: str = Query(..., description="YYYY-MM-DD")):
    """导出指定日期的打卡记录为 CSV 文件。"""
    export_path = os.path.join(settings.faces_dir, "..",
                               f"attendance_{date}.csv")
    attendance_service.export_csv(date, export_path)
    return FileResponse(
        export_path,
        media_type="text/csv",
        filename=f"attendance_{date}.csv",
    )


@app.get("/api/attendance/daily-counts")
async def api_daily_counts():
    """获取近 7 天每日打卡人数。"""
    end = datetime.now()
    start = end - timedelta(days=6)
    with db.get_connection() as conn:
        counts = db.get_daily_counts(conn,
                                      start.strftime("%Y-%m-%d"),
                                      end.strftime("%Y-%m-%d"))
    # Fill in missing days with zero
    counts_dict = {c["date"]: c["count"] for c in counts}
    result = []
    for i in range(7):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        result.append({"date": d, "count": counts_dict.get(d, 0)})
    return {"success": True, "counts": result}


# ---------------------------------------------------------------------------
# API - Stats
# ---------------------------------------------------------------------------
@app.get("/api/stats")
async def api_stats():
    """获取仪表盘统计数据。"""
    stats = attendance_service.get_dashboard_stats()
    return {
        "success": True,
        "stats": {
            "total_employees": stats.total_employees,
            "today_attendance": stats.today_attendance,
            "today_rate": stats.today_rate,
            "monthly_attendance": stats.monthly_attendance,
        },
    }


# ---------------------------------------------------------------------------
# API - Model training
# ---------------------------------------------------------------------------
@app.post("/api/train")
async def api_train():
    """触发模型重新训练。"""
    try:
        recognizer.train()
        recognizer.load_model()
        attendance_service.recognizer = recognizer
        # Restart camera if running
        cam = get_camera()
        if cam.has_frame():
            cam.restart()
        return {"success": True, "message": "模型训练成功"}
    except Exception as e:
        logger.exception("训练失败")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Video streaming (MJPEG)
# ---------------------------------------------------------------------------
@app.get("/video_feed")
async def video_feed():
    """MJPEG 视频流 - 实时人脸识别叠加显示。"""

    def generate():
        cam = get_camera()
        if not cam.start():
            # 摄像头不可用时，返回一张中文提示图片
            import cv2
            import numpy as np
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "摄像头不可用", (180, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(img, "请检查摄像头连接和 config.yaml 配置",
                        (140, 260), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            jpeg = encode_frame_jpeg(img)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )
            return

        try:
            while True:
                frame = cam.get_frame()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                        b"\r\n" + frame + b"\r\n"
                    )
                import time
                time.sleep(0.03)  # ~30 FPS
        except GeneratorExit:
            logger.info("Video feed client disconnected")
        finally:
            # Don't stop camera here - other clients may still be connected
            pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/events")
async def api_events():
    """获取最近的识别事件（前端轮询）。"""
    cam = get_camera()
    events = cam.get_events()
    return {"success": True, "events": events}


@app.post("/api/camera/start")
async def api_camera_start():
    """启动摄像头。"""
    cam = get_camera()
    success = cam.start()
    return {"success": success}


@app.post("/api/camera/stop")
async def api_camera_stop():
    """停止摄像头。"""
    cam = get_camera()
    cam.stop()
    return {"success": True}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "camera": _camera is not None and _camera.has_frame(),
    }


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown():
    global _camera
    if _camera:
        _camera.stop()
