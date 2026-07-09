"""
存储后端工厂
根据配置创建对应的存储后端实例
"""

from src.storage.base import StorageBackend
from src.storage.sqlite_backend import SQLiteBackend


def create_storage_backend(backend_type: str = "sqlite", **kwargs) -> StorageBackend:
    """工厂函数：根据类型创建存储后端实例

    参数:
        backend_type: 后端类型，当前支持 "sqlite"
        **kwargs: 传递给具体后端的参数（如 db_path）

    返回:
        StorageBackend 实例

    异常:
        ValueError: 如果指定的后端类型不支持
    """
    if backend_type == "sqlite":
        db_path = kwargs.get("db_path", "data/sqlite/chatbot.db")
        return SQLiteBackend(db_path=db_path)

    raise ValueError(
        f"不支持的存储后端类型: '{backend_type}'，当前仅支持: sqlite"
    )
