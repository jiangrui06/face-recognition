# 人脸识别打卡系统 v2.0

基于 **OpenCV** + **FastAPI** + **SQLite** 的全栈人脸识别考勤系统。支持实时人脸检测识别、Web 仪表盘视频流、考勤报表和员工管理。

## 功能特性

| 功能 | 说明 |
|------|------|
| **实时人脸识别** | LBPH 算法 + 摄像头实时检测，自动识别打卡 |
| **Web 仪表盘** | FastAPI 构建，支持 MJPEG 实时视频流 |
| **员工管理** | 注册新员工，多角度人脸样本采集 |
| **考勤追踪** | 识别后自动打卡，每日防重复 |
| **统计报表** | Chart.js 图表展示，支持 CSV 导出 |
| **双界面** | CLI 命令行 + Web 界面，双模式可用 |
| **DNN 检测** | 可选 OpenCV DNN (SSD) 检测，比 Haar 更准 |
| **SQLite 数据库** | 规范化关系表设计，支持索引和事务 |
| **容器化部署** | Docker + docker-compose 一键部署 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.9+, FastAPI, Uvicorn |
| 前端 | Bootstrap 5, HTMX, Chart.js |
| 数据库 | SQLite（WAL 模式、外键、索引） |
| 人脸检测 | Haar Cascade + DNN (Caffe SSD) |
| 人脸识别 | OpenCV LBPH（局部二值模式直方图） |
| 测试 | pytest |
| 运维 | Docker, docker-compose |

## 系统架构

```
face-recognition/
├── src/                         # 核心 Python 包
│   ├── main.py                 # CLI 命令行入口
│   ├── config.py               # YAML 配置 + 环境变量覆盖
│   ├── database.py             # SQLite 数据库层
│   ├── models.py               # 数据模型 (dataclass)
│   ├── face_detector.py        # 策略模式: Haar \| DNN 检测器
│   ├── face_recognizer.py      # LBPH 训练与识别
│   ├── face_register.py        # 人脸注册（含模糊检测）
│   ├── attendance.py           # 打卡业务逻辑与报表
│   ├── exceptions.py           # 自定义异常层次
│   └── utils.py                # 工具函数
├── web/                         # FastAPI Web 应用
│   ├── app.py                  # 路由 + REST API + MJPEG 流
│   ├── camera.py               # 后台摄像头管理线程
│   ├── templates/              # Jinja2 页面模板
│   └── static/                 # CSS + JS
├── tests/                       # pytest 测试套件
├── config.yaml                  # 配置文件
├── Dockerfile + docker-compose.yml
└── requirements.txt
```

## 安装运行

```bash
# 1. 进入项目目录
cd face-recognition

# 2. 安装依赖
pip install -r requirements.txt

# 3. 注册员工
python -m src.main --register

# 4. 训练模型
python -m src.main --train

# 5. 启动 Web 管理界面
python -m src.main --web
# 浏览器访问 http://localhost:8000

# 或者使用 CLI 菜单
python -m src.main
```

### Docker 部署

```bash
docker-compose up --build
```

## Web 页面

| 页面 | 说明 |
|------|------|
| **仪表盘** | 总览：员工数、今日打卡、出勤率、近7日趋势图 |
| **实时打卡** | MJPEG 视频流 + 实时识别结果叠加 + 事件日志 |
| **员工管理** | 查看/注册/删除员工 |
| **统计报表** | 日期范围查询、员工出勤明细、CSV 导出 |

## 运行测试

```bash
pytest tests/ -v
```

## 面试亮点

- **整洁架构**: 分层设计（配置 → 数据库 → 业务逻辑 → Web/CLI），单一职责
- **设计模式**: 策略模式（检测器）、工厂模式、单例模式（配置）、仓库模式（数据库）
- **数据库设计**: 规范化表结构、WAL 模式、外键约束、复合索引、软删除
- **Web 全栈**: FastAPI 异步接口、MJPEG 视频流、Jinja2 模板、HTMX 动态交互
- **代码质量**: 全量类型注解、34 个 pytest 测试、自定义异常层次、日志轮转
- **算法工程**: LBPH 人脸识别 + DNN 检测可选、图像质量过滤（模糊检测）
- **DevOps**: 多阶段 Docker 构建、docker-compose 编排、12-Factor 配置管理

## 许可证

MIT
