"""Configuration management - YAML + environment variable overrides."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Project root detection (walk up from this file to find project root)
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """Find project root by looking for config.yaml upwards."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT:
        return _PROJECT_ROOT

    current = Path(__file__).resolve().parent.parent
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            _PROJECT_ROOT = parent
            return parent

    # Fallback to cwd
    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


@dataclass
class Settings:
    """Application settings loaded from config.yaml + env overrides.

    Environment variables take precedence: prefix with ``FACE_``,
    using ``__`` as nested separator. Example:
    ``FACE_DATABASE__PATH=/tmp/db.sqlite`` overrides database.path.
    """
    # Database
    db_path: str = "data/face_recognition.db"

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Recognition
    confidence_threshold: int = 80
    cooldown_seconds: int = 5
    face_sample_count: int = 60
    face_min_size: int = 80

    # Detector
    detector_type: str = "haar"  # "haar" | "dnn"
    dnn_model_url: str = ""
    dnn_config_url: str = ""

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_max_bytes: int = 10_485_760
    log_backup_count: int = 5

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_reload: bool = False

    def resolve_paths(self, root: Path) -> None:
        """Convert relative paths to absolute paths."""
        self.db_path = str(root / self.db_path)
        self.log_file = str(root / self.log_file)

    @property
    def trainer_file(self) -> str:
        """Path to trainer.yml relative to project root."""
        return str(get_project_root() / "data" / "trainer.yml")

    @property
    def faces_dir(self) -> str:
        return str(get_project_root() / "data" / "faces")

    @property
    def model_cache_dir(self) -> str:
        return str(get_project_root() / "data" / "models")


def _load_config(path: Path) -> dict:
    """Load YAML config file, return empty dict if not found."""
    if not path.exists():
        logger.warning("Config file not found: %s, using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "__") -> dict:
    """Flatten nested dict for env var matching. e.g. {'a': {'b': 1}} -> {'a__b': 1}."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _apply_env_overrides(flat_config: dict, prefix: str = "FACE_") -> None:
    """Override config values from environment variables."""
    for key in list(flat_config.keys()):
        env_key = f"{prefix}{key.upper()}"
        if env_key in os.environ:
            flat_config[key] = os.environ[env_key]


def _map_to_settings(flat: dict) -> Settings:
    """Map flattened config keys to Settings fields."""
    key_map = {
        "database__path": "db_path",
        "camera__index": "camera_index",
        "camera__width": "camera_width",
        "camera__height": "camera_height",
        "camera__fps": "camera_fps",
        "recognition__confidence_threshold": "confidence_threshold",
        "recognition__cooldown_seconds": "cooldown_seconds",
        "recognition__face_sample_count": "face_sample_count",
        "recognition__face_min_size": "face_min_size",
        "detector__type": "detector_type",
        "detector__dnn__model_url": "dnn_model_url",
        "detector__dnn__config_url": "dnn_config_url",
        "logging__level": "log_level",
        "logging__file": "log_file",
        "logging__max_bytes": "log_max_bytes",
        "logging__backup_count": "log_backup_count",
        "server__host": "server_host",
        "server__port": "server_port",
        "server__reload": "server_reload",
    }

    kwargs = {}
    for config_key, field_name in key_map.items():
        val = flat.get(config_key)
        if val is not None:
            kwargs[field_name] = val

    return Settings(**kwargs)


_settings_cache: Optional[Settings] = None


def get_settings() -> Settings:
    """Get Settings singleton, loading from config.yaml and env vars."""
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    root = get_project_root()
    config_path = root / "config.yaml"

    raw = _load_config(config_path)
    flat = _flatten_dict(raw)
    _apply_env_overrides(flat)
    settings = _map_to_settings(flat)
    settings.resolve_paths(root)

    _settings_cache = settings
    return settings


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)."""
    global _settings_cache
    _settings_cache = None
    return get_settings()
