"""CLI entry point - interactive menu and command-line interface.

Usage:
    python -m src.main            # Interactive menu
    python -m src.main --register  # Quick register
    python -m src.main --train     # Quick train
    python -m src.main --web       # Start web server
"""

import argparse
import logging
import os
import sys
from typing import Optional

import cv2

from src.attendance import AttendanceService
from src.config import get_settings
from src.database import DatabaseManager
from src.face_recognizer import FaceRecognizer
from src.face_register import FaceRegister
from src.utils import setup_logging

logger = setup_logging("face_recognition")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str = "人脸识别打卡系统 v2.0") -> None:
    print("=" * 60)
    print(f"        {title}")
    print("=" * 60)


def print_menu() -> None:
    print("\n  主菜单:")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  1. 📷 实时人脸识别打卡                      │")
    print("  │  2. 👤 注册新员工                            │")
    print("  │  3. 🎯 训练识别模型                          │")
    print("  │  4. 📋 查看今日打卡记录                      │")
    print("  │  5. 👥 已注册员工列表                        │")
    print("  │  6. 📊 查看统计报表                          │")
    print("  │  7. 🌐 启动 Web 管理界面                     │")
    print("  │  0. 🚪 退出系统                              │")
    print("  └─────────────────────────────────────────────┘")


def cli_attendance() -> None:
    """Run real-time face recognition attendance."""
    recognizer = FaceRecognizer()
    if not recognizer.load_model():
        print("\n错误: 未找到训练模型，请先注册员工并训练模型。")
        print("请执行: python -m src.main --register 和 python -m src.main --train")
        input("\n按 Enter 返回...")
        return

    attendance = AttendanceService()
    cap = cv2.VideoCapture(get_settings().camera_index)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        input("\n按 Enter 返回...")
        return

    print("\n人脸识别打卡系统已启动")
    print("=" * 40)
    print(f"{'姓名':<12} {'状态':<12} {'时间'}")
    print("-" * 40)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process every other frame for performance
        if frame_count % 2 == 0:
            annotated_frame, events = attendance.process_frame(frame)
        else:
            annotated_frame = frame

        # Print events
        now = cv2.waitKey(1) & 0xFF
        for event in events:
            ts = __import__("datetime").datetime.now().strftime("%H:%M:%S")
            icon = "✓" if event["checked_in"] else "○"
            print(f"{event['name']:<12} {icon} 已打卡    {ts}")

        # Draw UI
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "Face Recognition Attendance",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.rectangle(annotated_frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated_frame, "ESC: Exit  R: Records  T: Retrain",
                    (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        cv2.imshow("Face Recognition Attendance", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("r") or key == ord("R"):
            records = attendance.get_today_records()
            print(f"\n今日打卡记录 ({len(records)}人):")
            for r in records:
                t = r.check_in.strftime("%H:%M:%S")
                print(f"  {r.employee_name:<12} {t}")
        elif key == ord("t") or key == ord("T"):
            cap.release()
            cv2.destroyAllWindows()
            try:
                recognizer.train()
                print("模型已重新训练")
            except Exception as e:
                print(f"训练失败: {e}")
            cap = cv2.VideoCapture(get_settings().camera_index)

    cap.release()
    cv2.destroyAllWindows()
    print("\n打卡系统已关闭。")


def cli_register() -> None:
    """Register a new employee via CLI."""
    name = input("请输入员工姓名: ").strip()
    if not name:
        print("姓名不能为空！")
        return

    register = FaceRegister()
    try:
        result = register.register(name)
        print(f"\n{'✓' if result['success'] else '✗'} {result['message']}")
    except Exception as e:
        print(f"\n✗ 注册失败: {e}")


def cli_train() -> None:
    """Train the recognition model."""
    recognizer = FaceRecognizer()
    try:
        count = recognizer.train()
        print(f"\n✓ 模型训练完成！使用了 {count} 个样本。")
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")


def cli_today() -> None:
    """Show today's attendance records."""
    attendance = AttendanceService()
    records = attendance.get_today_records()
    if not records:
        print("\n今日暂无打卡记录。")
        return

    print(f"\n今日打卡记录 ({len(records)}人):")
    print("-" * 40)
    print(f"{'姓名':<12} {'时间':<10} {'置信度':<10}")
    print("-" * 40)
    for r in records:
        t = r.check_in.strftime("%H:%M:%S") if isinstance(
            r.check_in, object) else str(r.check_in)[11:19]
        conf = f"{r.confidence:.1f}" if r.confidence else "-"
        print(f"{r.employee_name:<12} {t:<10} {conf:<10}")


def cli_employees() -> None:
    """List all registered employees."""
    register = FaceRegister()
    employees = register.list_employees()
    if not employees:
        print("\n当前没有已注册的员工。")
        return

    print(f"\n已注册员工 ({len(employees)}人):")
    print("-" * 50)
    print(f"{'ID':<5} {'姓名':<12} {'样本数':<8} {'注册时间'}")
    print("-" * 50)
    for emp in employees:
        ts = emp["created_at"][:19] if isinstance(
            emp["created_at"], str) else str(emp["created_at"])[:19]
        print(f"{emp['id']:<5} {emp['name']:<12} "
              f"{emp.get('sample_count', 0):<8} {ts}")


def cli_stats() -> None:
    """Show attendance statistics."""
    attendance = AttendanceService()
    stats = attendance.get_dashboard_stats()
    print(f"\n{'='*50}")
    print("  打卡统计")
    print(f"{'='*50}")
    print(f"  总员工数:     {stats.total_employees}")
    print(f"  今日打卡:     {stats.today_attendance}")
    print(f"  出勤率:       {stats.today_rate}%")
    print(f"  本月打卡:     {stats.monthly_attendance}")
    print(f"{'='*50}")


def cli_web() -> None:
    """Start the web server."""
    print("\n正在启动 Web 管理界面...")
    print(f"访问地址: http://localhost:{get_settings().server_port}")
    print("按 Ctrl+C 停止服务器\n")

    import uvicorn
    from web.app import app
    uvicorn.run(
        app,
        host=get_settings().server_host,
        port=get_settings().server_port,
        log_level="info",
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="人脸识别打卡系统 v2.0"
    )
    parser.add_argument("--register", action="store_true",
                        help="注册新员工")
    parser.add_argument("--train", action="store_true",
                        help="训练识别模型")
    parser.add_argument("--attendance", action="store_true",
                        help="启动实时打卡")
    parser.add_argument("--web", action="store_true",
                        help="启动 Web 管理界面")
    parser.add_argument("--migrate", action="store_true",
                        help="从旧版 CSV 迁移数据")

    args = parser.parse_args()

    # Initialize database
    db = DatabaseManager()
    db.init_db()

    # Quick commands
    if args.register:
        cli_register()
        return
    elif args.train:
        cli_train()
        return
    elif args.attendance:
        cli_attendance()
        return
    elif args.web:
        cli_web()
        return
    elif args.migrate:
        old_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "face recognition"
        )
        names_file = os.path.join(old_dir, "data", "names.txt")
        logs_dir = os.path.join(old_dir, "attendance_logs")
        result = db.migrate_from_legacy(names_file, logs_dir)
        print(f"\n数据迁移完成: {result['employees']} 员工, {result['records']} 记录")
        return

    # Interactive menu
    while True:
        clear_screen()
        print_header()
        print_menu()
        choice = input("\n  请选择操作: ").strip()

        if choice == "1":
            clear_screen()
            print_header("实时人脸识别打卡")
            cli_attendance()
            input("\n按 Enter 返回主菜单...")
        elif choice == "2":
            clear_screen()
            print_header("注册新员工")
            cli_register()
            input("\n按 Enter 返回主菜单...")
        elif choice == "3":
            clear_screen()
            print_header("训练识别模型")
            cli_train()
            input("\n按 Enter 返回主菜单...")
        elif choice == "4":
            clear_screen()
            print_header("今日打卡记录")
            cli_today()
            input("\n按 Enter 返回主菜单...")
        elif choice == "5":
            clear_screen()
            print_header("已注册员工列表")
            cli_employees()
            input("\n按 Enter 返回主菜单...")
        elif choice == "6":
            clear_screen()
            print_header("统计报表")
            cli_stats()
            input("\n按 Enter 返回主菜单...")
        elif choice == "7":
            cli_web()
        elif choice == "0":
            clear_screen()
            print("\n感谢使用人脸识别打卡系统，再见！\n")
            sys.exit(0)
        else:
            print("\n  无效输入，请重新选择。")
            input("按 Enter 继续...")


if __name__ == "__main__":
    main()
