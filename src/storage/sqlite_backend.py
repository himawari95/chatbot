"""
SQLite 存储后端实现（基于 SQLAlchemy + aiosqlite）
包含 ORM 模型定义及所有 CRUD 操作的具体实现
"""

import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from src.storage.base import StorageBackend

logger = logging.getLogger("chatbot")


# =============================================================================
# SQLAlchemy ORM 模型定义
# =============================================================================


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


class UserModel(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    sessions = relationship("SessionModel", back_populates="user")


class SessionModel(Base):
    """会话表"""
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True)
    title = Column(String(200), default="")
    role = Column(String(20), default="default", server_default="default")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    user = relationship("UserModel", back_populates="sessions")
    messages = relationship(
        "MessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class MessageModel(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), nullable=False)      # "user" 或 "assistant"
    role_name = Column(String(20), default="default", server_default="default")
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    session = relationship("SessionModel", back_populates="messages")


# =============================================================================
# SQLite 存储后端实现
# =============================================================================


class SQLiteBackend(StorageBackend):
    """基于 SQLite 的存储后端，使用 aiosqlite 异步驱动"""

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        database_url = f"sqlite+aiosqlite:///{self._db_path.as_posix()}"
        self._engine = create_async_engine(
            database_url, echo=False, connect_args={"check_same_thread": False}
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """创建表结构并执行必要的迁移"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查数据库文件是否损坏
        if self._db_path.exists():
            try:
                async with self._engine.connect() as conn:
                    await conn.get_raw_connection()
            except Exception:
                self._db_path.unlink(missing_ok=True)

        # 迁移：检查是否需要重建或添加列
        if self._db_path.exists():
            import sqlite3 as _sync_sqlite
            conn = _sync_sqlite.connect(str(self._db_path))

            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchall()
            if not tables:
                conn.close()
                logger.info("检测到旧数据库结构，重建中...")
                self._db_path.unlink()
            else:
                # 为 sessions 表添加 role 列（如果缺少）
                cols = [r[1] for r in conn.execute("PRAGMA table_info('sessions')").fetchall()]
                conn.close()
                if "role" not in cols:
                    logger.info("为 sessions 表添加 'role' 列")
                    c2 = _sync_sqlite.connect(str(self._db_path))
                    c2.execute("ALTER TABLE sessions ADD COLUMN role VARCHAR(20) DEFAULT 'default'")
                    c2.commit()
                    c2.close()

                # 为 messages 表添加 role_name 列（如果缺少）
                c2 = _sync_sqlite.connect(str(self._db_path))
                msg_cols = [r[1] for r in c2.execute("PRAGMA table_info('messages')").fetchall()]
                c2.close()
                if "role_name" not in msg_cols:
                    logger.info("为 messages 表添加 'role_name' 列")
                    c3 = _sync_sqlite.connect(str(self._db_path))
                    c3.execute("ALTER TABLE messages ADD COLUMN role_name VARCHAR(20) DEFAULT 'default'")
                    c3.commit()
                    c3.close()

        # 创建所有表
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """释放数据库引擎资源"""
        await self._engine.dispose()

    # ------------------------------------------------------------------
    # 用户操作
    # ------------------------------------------------------------------

    async def get_user(self, username: str) -> dict | None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(UserModel).where(UserModel.username == username)
            )
            user = result.scalar_one_or_none()
        if not user:
            return None
        return {
            "id": user.id,
            "username": user.username,
            "created_at": str(user.created_at),
        }

    async def get_or_create_user(self, username: str) -> dict:
        async with self._session_factory() as db:
            result = await db.execute(
                select(UserModel).where(UserModel.username == username)
            )
            user = result.scalar_one_or_none()
            if not user:
                user = UserModel(username=username)
                db.add(user)
                await db.commit()
                await db.refresh(user)
        return {
            "id": user.id,
            "username": user.username,
            "created_at": str(user.created_at),
        }

    # ------------------------------------------------------------------
    # 会话操作
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> dict | None:
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(SessionModel).where(SessionModel.id == session_id)
                )
            ).scalar_one_or_none()
        if not row:
            return None
        return {
            "id": row.id,
            "title": row.title,
            "role": row.role,
            "user_id": row.user_id,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
        }

    async def get_sessions_for_user(self, user_id: int) -> list[dict]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(SessionModel)
                .where(SessionModel.user_id == user_id)
                .order_by(SessionModel.updated_at.desc())
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "role": r.role,
                "created_at": str(r.created_at),
                "updated_at": str(r.updated_at),
            }
            for r in rows
        ]

    async def create_session(
        self, session_id: str, user_id: int, title: str = "", role: str = "default"
    ) -> None:
        async with self._session_factory() as db:
            db.add(SessionModel(id=session_id, title=title, user_id=user_id, role=role))
            await db.commit()

    async def delete_session(self, session_id: str) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(SessionModel).where(SessionModel.id == session_id))
            await db.commit()

    async def rename_session(self, session_id: str, title: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(title=title, updated_at=func.now())
            )
            await db.commit()

    async def update_session_role(self, session_id: str, role: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(role=role, updated_at=func.now())
            )
            await db.commit()

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    async def get_messages_for_role(
        self, session_id: str, role_name: str
    ) -> list[dict]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(MessageModel)
                .where(
                    MessageModel.session_id == session_id,
                    MessageModel.role_name == role_name,
                )
                .order_by(MessageModel.timestamp.asc())
            )
            rows = result.scalars().all()
        return [{"role": r.role, "content": r.content} for r in rows]

    async def save_message(
        self, session_id: str, role: str, content: str, role_name: str = "default"
    ) -> None:
        async with self._session_factory() as db:
            db.add(
                MessageModel(
                    session_id=session_id,
                    role=role,
                    content=content,
                    role_name=role_name,
                )
            )
            await db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(updated_at=func.now())
            )
            await db.commit()

    async def auto_title(self, session_id: str, message: str) -> None:
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(SessionModel).where(SessionModel.id == session_id)
                )
            ).scalar_one_or_none()
            if row and not row.title:
                row.title = message[:20]
                await db.commit()

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.get_raw_connection()
            return True
        except Exception:
            return False
