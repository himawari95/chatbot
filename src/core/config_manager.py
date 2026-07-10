"""
配置管理器
负责加载和管理所有配置来源：.env 环境变量、config.yaml 全局配置、config/logging.yaml 日志配置
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# =============================================================================
# JSON 日志格式化器
# =============================================================================


class JsonFormatter(logging.Formatter):
    """将日志记录输出为 JSON 格式，便于采集和分析"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 附加上下文字段（通过 extra 传入）
        for key in ("user_id", "session_id", "duration", "operation", "model"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        # 异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


# =============================================================================
# 配置管理器
# =============================================================================


class ConfigManager:
    """统一配置管理器，聚合 .env + YAML 配置"""

    def __init__(self, config_dir: str = "config", env_file: str = ".env"):
        self._config_dir = Path(config_dir)
        self._env_file = Path(env_file)

        # 加载 .env 文件
        load_dotenv(self._env_file)

        # 加载 YAML 配置
        self._global_config: dict[str, Any] = {}
        self._logging_config: dict[str, Any] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # 加载阶段
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """加载所有 YAML 配置文件"""
        self._load_global_config()
        self._load_logging_config()
        self._setup_logging()

    def _load_global_config(self) -> None:
        """加载 config.yaml 全局配置"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._global_config = yaml.safe_load(f) or {}

    def _load_logging_config(self) -> None:
        """加载 config/logging.yaml 日志配置"""
        logging_path = self._config_dir / "logging.yaml"
        if logging_path.exists():
            with open(logging_path, "r", encoding="utf-8") as f:
                self._logging_config = yaml.safe_load(f) or {}

    def _setup_logging(self) -> None:
        """程序化配置日志系统：JSON 格式 + 控制台 + 文件滚动"""
        if not self._logging_config:
            self._setup_default_logging()
            return

        log_level = self._logging_config.get("level", "INFO").upper()
        use_json = self._logging_config.get("json_format", True)
        outputs = self._logging_config.get("outputs", {})

        # 根 logger
        root = logging.getLogger()
        root.setLevel(getattr(logging, log_level, logging.INFO))

        # 选择格式化器
        if use_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # 控制台输出
        console_cfg = outputs.get("console", {})
        if console_cfg.get("enabled", True):
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(getattr(logging, console_cfg.get("level", "INFO").upper(), logging.INFO))
            ch.setFormatter(formatter)
            root.addHandler(ch)

        # 文件输出（按天滚动）
        file_cfg = outputs.get("file", {})
        if file_cfg.get("enabled", True):
            file_path = file_cfg.get("path", "logs/app.log")
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.TimedRotatingFileHandler(
                filename=file_path,
                when=file_cfg.get("when", "midnight"),
                backupCount=file_cfg.get("backup_count", 7),
                encoding=file_cfg.get("encoding", "utf-8"),
            )
            fh.setLevel(getattr(logging, file_cfg.get("level", "DEBUG").upper(), logging.DEBUG))
            fh.setFormatter(formatter)
            root.addHandler(fh)

        # 抑制第三方库日志噪音
        for lib in self._logging_config.get("quiet_libs", []):
            logging.getLogger(lib).setLevel(logging.WARNING)

    def _setup_default_logging(self) -> None:
        """回退日志配置（仅控制台）"""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def init_logging(self) -> None:
        """供 main.py 在启动时调用，确保日志目录和文件就绪"""
        log_path = self._logging_config.get("outputs", {}).get("file", {}).get("path", "logs/app.log")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 环境变量访问
    # ------------------------------------------------------------------

    def get_env(self, key: str, default: str = "") -> str:
        """获取环境变量"""
        return os.getenv(key, default)

    # ------------------------------------------------------------------
    # 服务器配置
    # ------------------------------------------------------------------

    @property
    def server_host(self) -> str:
        return self._global_config.get("server", {}).get("host", "127.0.0.1")

    @property
    def server_port(self) -> int:
        return self._global_config.get("server", {}).get("port", 8000)

    # ------------------------------------------------------------------
    # 数据库配置
    # ------------------------------------------------------------------

    @property
    def db_type(self) -> str:
        return self._global_config.get("database", {}).get("type", "sqlite")

    @property
    def db_path(self) -> str:
        return self._global_config.get("database", {}).get("path", "data/sqlite/chatbot.db")

    # ------------------------------------------------------------------
    # 模型配置
    # ------------------------------------------------------------------

    @property
    def models(self) -> list[dict]:
        return self._global_config.get("models", [])

    @property
    def default_model(self) -> str:
        return self._global_config.get("llm", {}).get("default_model", "deepseek-chat")

    @property
    def llm_temperature(self) -> float:
        return self._global_config.get("llm", {}).get("temperature", 0.7)

    @property
    def llm_max_tokens(self) -> int:
        return self._global_config.get("llm", {}).get("max_tokens", 2048)

    @property
    def llm_timeout(self) -> int:
        """LLM 调用超时时间（秒）"""
        return self._global_config.get("llm", {}).get("timeout", 30)

    @property
    def llm_max_retries(self) -> int:
        """LLM 调用最大重试次数"""
        return self._global_config.get("llm", {}).get("max_retries", 3)

    @property
    def llm_retry_backoff(self) -> int:
        """LLM 重试退避因子"""
        return self._global_config.get("llm", {}).get("retry_backoff", 2)

    def get_model_config(self, model_name: str) -> dict | None:
        """根据模型名称获取对应的 API 配置"""
        for m in self.models:
            if m.get("name") == model_name:
                api_key_env = m.get("api_key_env", "")
                base_url_env = m.get("base_url_env", "")
                return {
                    "api_key": self.get_env(api_key_env),
                    "base_url": self.get_env(base_url_env),
                }
        return None

    # ------------------------------------------------------------------
    # 多模态 / 视觉模型配置
    # ------------------------------------------------------------------

    @property
    def vision_enabled(self) -> bool:
        return self._global_config.get("vision", {}).get("enabled", True)

    @property
    def vision_model(self) -> str:
        return self._global_config.get("vision", {}).get("model", "qwen-vl-plus")

    @property
    def vision_max_file_size_mb(self) -> int:
        return self._global_config.get("vision", {}).get("max_file_size_mb", 10)

    def get_vision_model_config(self) -> dict | None:
        """获取视觉模型配置（API Key + Base URL）"""
        vision = self._global_config.get("vision", {})
        provider = vision.get("provider", "dashscope")
        if provider == "dashscope":
            return {
                "api_key": self.get_env("DASHSCOPE_API_KEY"),
                "base_url": self.get_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                "model": vision.get("model", "qwen-vl-plus"),
            }
        return None

    # ------------------------------------------------------------------
    # UI 配置
    # ------------------------------------------------------------------

    @property
    def ui_host(self) -> str:
        return self._global_config.get("ui", {}).get("host", "127.0.0.1")

    @property
    def ui_port(self) -> int:
        return self._global_config.get("ui", {}).get("port", 7860)

    @property
    def ui_share(self) -> bool:
        return self._global_config.get("ui", {}).get("share", False)


# 全局配置单例
_config_instance: ConfigManager | None = None


def get_config() -> ConfigManager:
    """获取全局配置管理器单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance
