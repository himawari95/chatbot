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
    model_name = Column(String(50), default="deepseek-chat", server_default="deepseek-chat")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_prompt_tokens = Column(Integer, default=0)
    total_completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    user = relationship("UserModel", back_populates="sessions")
    messages = relationship(
        "MessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class PresetModel(Base):
    """预设表 — 用户自定义角色预设"""
    __tablename__ = "presets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(50), nullable=False)
    description = Column(String(200), default="")
    system_prompt = Column(Text, nullable=False)
    is_builtin = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


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
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
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
                logger.warning("数据库文件损坏，正在重建", extra={"operation": "db_init", "db_path": str(self._db_path)})
                self._db_path.unlink(missing_ok=True)

        # 迁移：检查是否需要重建或添加列（使用同一个同步连接完成所有操作）
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
                if "role" not in cols:
                    logger.info("为 sessions 表添加 'role' 列")
                    conn.execute("ALTER TABLE sessions ADD COLUMN role VARCHAR(20) DEFAULT 'default'")

                # 为 messages 表添加 role_name 列（如果缺少）
                msg_cols = [r[1] for r in conn.execute("PRAGMA table_info('messages')").fetchall()]
                if "role_name" not in msg_cols:
                    logger.info("为 messages 表添加 'role_name' 列")
                    conn.execute("ALTER TABLE messages ADD COLUMN role_name VARCHAR(20) DEFAULT 'default'")

                # 为 sessions 表添加 model_name 列（如果缺少）
                sess_cols = [r[1] for r in conn.execute("PRAGMA table_info('sessions')").fetchall()]
                if "model_name" not in sess_cols:
                    logger.info("为 sessions 表添加 'model_name' 列")
                    conn.execute("ALTER TABLE sessions ADD COLUMN model_name VARCHAR(50) DEFAULT 'deepseek-chat'")

                # 为 messages 表添加 token 列（如果缺少）
                msg_cols2 = [r[1] for r in conn.execute("PRAGMA table_info('messages')").fetchall()]
                if "prompt_tokens" not in msg_cols2:
                    logger.info("为 messages 表添加 token 列")
                    conn.execute("ALTER TABLE messages ADD COLUMN prompt_tokens INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE messages ADD COLUMN completion_tokens INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE messages ADD COLUMN total_tokens INTEGER DEFAULT 0")

                # 为 sessions 表添加 token 统计列（如果缺少）
                sess_cols2 = [r[1] for r in conn.execute("PRAGMA table_info('sessions')").fetchall()]
                if "total_prompt_tokens" not in sess_cols2:
                    logger.info("为 sessions 表添加 token 统计列")
                    conn.execute("ALTER TABLE sessions ADD COLUMN total_prompt_tokens INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE sessions ADD COLUMN total_completion_tokens INTEGER DEFAULT 0")
                    conn.execute("ALTER TABLE sessions ADD COLUMN total_tokens INTEGER DEFAULT 0")

                # 检查并创建 presets 表
                preset_tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='presets'"
                ).fetchall()
                if not preset_tables:
                    logger.info("创建 presets 表...")
                    conn.execute("""
                        CREATE TABLE presets (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            name VARCHAR(50) NOT NULL,
                            description VARCHAR(200) DEFAULT '',
                            system_prompt TEXT NOT NULL,
                            is_builtin INTEGER DEFAULT 0,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                conn.commit()
                conn.close()

        # 创建所有表（SQLAlchemy ORM 管理的表）
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

    async def delete_user(self, user_id: int) -> None:
        """删除用户及所有关联的会话、消息和预设（级联删除）"""
        async with self._session_factory() as db:
            await db.execute(delete(PresetModel).where(PresetModel.user_id == user_id))
            await db.execute(delete(SessionModel).where(SessionModel.user_id == user_id))
            await db.execute(delete(UserModel).where(UserModel.id == user_id))
            await db.commit()

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
            "model_name": row.model_name,
            "user_id": row.user_id,
            "total_prompt_tokens": row.total_prompt_tokens,
            "total_completion_tokens": row.total_completion_tokens,
            "total_tokens": row.total_tokens,
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
                "model_name": r.model_name,
                "total_prompt_tokens": r.total_prompt_tokens,
                "total_completion_tokens": r.total_completion_tokens,
                "total_tokens": r.total_tokens,
                "created_at": str(r.created_at),
                "updated_at": str(r.updated_at),
            }
            for r in rows
        ]

    async def create_session(
        self, session_id: str, user_id: int, title: str = "",
        role: str = "default", model_name: str = "deepseek-chat",
    ) -> None:
        async with self._session_factory() as db:
            db.add(SessionModel(
                id=session_id, title=title, user_id=user_id,
                role=role, model_name=model_name,
            ))
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

    async def update_session_model(self, session_id: str, model_name: str) -> None:
        async with self._session_factory() as db:
            await db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(model_name=model_name, updated_at=func.now())
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

    async def get_all_messages(self, session_id: str) -> list[dict]:
        """获取某会话的所有消息（不区分角色，按时间升序）"""
        async with self._session_factory() as db:
            result = await db.execute(
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.timestamp.asc())
            )
            rows = result.scalars().all()
        return [
            {"role": r.role, "content": r.content, "timestamp": str(r.timestamp)}
            for r in rows
        ]

    async def save_message(
        self, session_id: str, role: str, content: str, role_name: str = "default",
        prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0,
    ) -> None:
        """保存消息，可选记录 token 用量"""
        async with self._session_factory() as db:
            db.add(
                MessageModel(
                    session_id=session_id,
                    role=role,
                    content=content,
                    role_name=role_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
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
    # Token 统计
    # ------------------------------------------------------------------

    async def get_session_tokens(self, session_id: str) -> dict:
        """获取会话累计 token 用量"""
        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(SessionModel).where(SessionModel.id == session_id)
                )
            ).scalar_one_or_none()
        if not row:
            return {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0}
        return {
            "total_prompt_tokens": row.total_prompt_tokens,
            "total_completion_tokens": row.total_completion_tokens,
            "total_tokens": row.total_tokens,
        }

    async def update_session_tokens(self, session_id: str) -> None:
        """重新计算并更新会话的 token 累计值（汇总该会话所有消息）"""
        async with self._session_factory() as db:
            # 汇总 messages 表中该会话的所有 token
            result = await db.execute(
                select(
                    func.coalesce(func.sum(MessageModel.prompt_tokens), 0),
                    func.coalesce(func.sum(MessageModel.completion_tokens), 0),
                    func.coalesce(func.sum(MessageModel.total_tokens), 0),
                ).where(MessageModel.session_id == session_id)
            )
            p_sum, c_sum, t_sum = result.one()
            await db.execute(
                update(SessionModel)
                .where(SessionModel.id == session_id)
                .values(
                    total_prompt_tokens=p_sum,
                    total_completion_tokens=c_sum,
                    total_tokens=t_sum,
                )
            )
            await db.commit()

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    async def search_messages(self, user_id: int, keyword: str) -> list[dict]:
        """搜索用户所有会话中的消息（JOIN sessions 表，按用户过滤）"""
        async with self._session_factory() as db:
            result = await db.execute(
                select(
                    MessageModel.session_id,
                    SessionModel.title.label("session_title"),
                    MessageModel.content,
                    MessageModel.timestamp,
                )
                .join(SessionModel, MessageModel.session_id == SessionModel.id)
                .where(
                    SessionModel.user_id == user_id,
                    MessageModel.content.ilike(f"%{keyword}%"),
                )
                .order_by(MessageModel.timestamp.desc())
                .limit(50)
            )
            rows = result.all()
        return [
            {
                "session_id": r.session_id,
                "session_title": r.session_title,
                "content": r.content,
                "timestamp": str(r.timestamp),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 预设操作
    # ------------------------------------------------------------------

    async def create_preset(
        self, user_id: int, name: str, description: str, system_prompt: str
    ) -> dict:
        async with self._session_factory() as db:
            preset = PresetModel(
                user_id=user_id,
                name=name,
                description=description,
                system_prompt=system_prompt,
            )
            db.add(preset)
            await db.commit()
            await db.refresh(preset)
        return {
            "id": preset.id,
            "user_id": preset.user_id,
            "name": preset.name,
            "description": preset.description,
            "system_prompt": preset.system_prompt,
            "is_builtin": bool(preset.is_builtin),
            "created_at": str(preset.created_at),
            "updated_at": str(preset.updated_at),
        }

    async def get_presets_for_user(self, user_id: int) -> list[dict]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(PresetModel)
                .where(PresetModel.user_id == user_id)
                .order_by(PresetModel.created_at.desc())
            )
            rows = result.scalars().all()
        return [
            {
                "id": p.id,
                "user_id": p.user_id,
                "name": p.name,
                "description": p.description,
                "system_prompt": p.system_prompt,
                "is_builtin": bool(p.is_builtin),
                "created_at": str(p.created_at),
                "updated_at": str(p.updated_at),
            }
            for p in rows
        ]

    async def get_preset_by_name(self, name: str, user_id: int) -> dict | None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(PresetModel).where(
                    PresetModel.name == name,
                    PresetModel.user_id == user_id,
                )
            )
            preset = result.scalar_one_or_none()
        if not preset:
            return None
        return {
            "id": preset.id,
            "user_id": preset.user_id,
            "name": preset.name,
            "description": preset.description,
            "system_prompt": preset.system_prompt,
            "is_builtin": bool(preset.is_builtin),
            "created_at": str(preset.created_at),
            "updated_at": str(preset.updated_at),
        }

    async def update_preset(
        self, preset_id: int, user_id: int,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
    ) -> dict | None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(PresetModel).where(
                    PresetModel.id == preset_id,
                    PresetModel.user_id == user_id,
                )
            )
            preset = result.scalar_one_or_none()
            if not preset:
                return None
            if name is not None:
                preset.name = name
            if description is not None:
                preset.description = description
            if system_prompt is not None:
                preset.system_prompt = system_prompt
            preset.updated_at = func.now()
            await db.commit()
            await db.refresh(preset)
        return {
            "id": preset.id,
            "user_id": preset.user_id,
            "name": preset.name,
            "description": preset.description,
            "system_prompt": preset.system_prompt,
            "is_builtin": bool(preset.is_builtin),
            "created_at": str(preset.created_at),
            "updated_at": str(preset.updated_at),
        }

    async def delete_preset(self, preset_id: int, user_id: int) -> bool:
        async with self._session_factory() as db:
            result = await db.execute(
                select(PresetModel).where(
                    PresetModel.id == preset_id,
                    PresetModel.user_id == user_id,
                )
            )
            preset = result.scalar_one_or_none()
            if not preset:
                return False
            await db.delete(preset)
            await db.commit()
        return True

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
