"""
人脸识别打卡系统 - 主入口
"""
import os
import sys


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 55)
    print("       人脸识别打卡系统 v1.0")
    print("=" * 55)


def print_menu():
    print("\n  主菜单:")
    print("  ┌─────────────────────────────────────┐")
    print("  │  1. 📷 开始人脸识别打卡              │")
    print("  │  2. 👤 注册新员工                    │")
    print("  │  3. 🎯 训练识别模型                  │")
    print("  │  4. 📋 查看打卡记录                  │")
    print("  │  5. 👥 查看已注册员工                │")
    print("  │  0. 🚪 退出系统                      │")
    print("  └─────────────────────────────────────┘")


def main():
    while True:
        clear_screen()
        print_header()
        print_menu()
        choice = input("\n  请选择操作: ").strip()

        if choice == "1":
            clear_screen()
            print_header()
            print("\n  ▶ 启动人脸识别打卡...")
            from attendance import run_attendance
            run_attendance()
            input("\n按 Enter 返回主菜单...")

        elif choice == "2":
            clear_screen()
            print_header()
            print("\n  ▶ 注册新员工...")
            from face_register import register_face
            register_face()
            input("\n按 Enter 返回主菜单...")

        elif choice == "3":
            clear_screen()
            print_header()
            print("\n  ▶ 训练识别模型...")
            from train_model import train_model
            train_model()
            input("\n按 Enter 返回主菜单...")

        elif choice == "4":
            clear_screen()
            print_header()
            print("\n  ▶ 查看打卡记录...")
            from attendance import view_records
            view_records()
            input("\n按 Enter 返回主菜单...")

        elif choice == "5":
            clear_screen()
            print_header()
            print("\n  ▶ 已注册员工列表...")
            from face_register import list_registered
            list_registered()
            input("\n按 Enter 返回主菜单...")

        elif choice == "0":
            clear_screen()
            print("\n感谢使用人脸识别打卡系统，再见！\n")
            sys.exit(0)

        else:
            print("\n  无效输入，请重新选择。")
            input("按 Enter 继续...")


if __name__ == "__main__":
    main()
