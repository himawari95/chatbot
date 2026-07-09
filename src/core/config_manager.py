"""
配置管理器
负责加载和管理所有配置来源：.env 环境变量、config.yaml 全局配置、config/logging.yaml 日志配置
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigManager:
    """统一配置管理器，聚合 .env + YAML 配置"""

    def __init__(self, config_dir: str = "config", env_file: str = ".env"):
        self._config_dir = Path(config_dir)
        self._env_file = Path(env_file)

        # 加载 .env 文件
        load_dotenv(self._env_file)

        # 加载 YAML 配置
        self._global_config: dict[str, Any] = {}
        self._presets: list[dict] = []
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
        """应用日志配置"""
        if self._logging_config:
            # 确保 logs 目录存在
            for handler in self._logging_config.get("handlers", {}).values():
                filename = handler.get("filename", "")
                if filename:
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
            logging.config.dictConfig(self._logging_config)

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
