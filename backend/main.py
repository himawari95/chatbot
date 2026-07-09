"""
Chatbot Backend
FastAPI + LangChain ConversationChain + SQLAlchemy async (aiosqlite)
Multi-user chatbot with session memory, SSE streaming, and user isolation.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
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

load_dotenv()
logger = logging.getLogger("chatbot")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict = {
    "deepseek-chat": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
    },
}

# ---------------------------------------------------------------------------
# Role / Persona definitions
# ---------------------------------------------------------------------------
ROLE_PROMPTS: dict[str, str] = {
    "default": "你是一个友好的AI助手，请用中文回答用户的问题。",
    "teacher": "你是一位耐心的老师，善于用通俗易懂的方式讲解复杂概念。回答要详细、有条理，多用例子说明。",
    "programmer": "你是一位资深程序员，回答要简洁、技术导向。涉及代码时要给出完整的示例，注重代码质量。",
    "philosopher": "你是一位哲学家，善于深度思考。回答要富有哲理，引导用户思考问题的本质。",
    "friend": "你是一位贴心的朋友，回答要轻松、幽默、随和，像朋友聊天一样自然。",
}

# ---------------------------------------------------------------------------
# SQLAlchemy async setup (aiosqlite)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent.parent / "data" / "chatbot.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    sessions = relationship("SessionModel", back_populates="user")


class SessionModel(Base):
    __tablename__ = "sessions"
    id = Column(String(32), primary_key=True)
    title = Column(String(200), default="")
    role = Column(String(20), default="default", server_default="default")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "MessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class MessageModel(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), nullable=False)
    role_name = Column(String(20), default="default", server_default="default")
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    session = relationship("SessionModel", back_populates="messages")


async def _init_db() -> None:
    """Create tables and run migrations."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        try:
            async with engine.connect() as conn:
                await conn.get_raw_connection()
        except Exception:
            DB_PATH.unlink(missing_ok=True)
    # Check if users table exists (migration gate for v1 → v2)
    if DB_PATH.exists():
        import sqlite3 as _sync_sqlite
        _c = _sync_sqlite.connect(str(DB_PATH))
        _tables = _c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchall()
        if not _tables:
            _c.close()
            logger.info("Old schema detected — recreating database")
            DB_PATH.unlink()
        else:
            # Migration: add role column to sessions if missing
            _cols = [r[1] for r in _c.execute("PRAGMA table_info('sessions')").fetchall()]
            _c.close()
            if "role" not in _cols:
                logger.info("Adding 'role' column to sessions table")
                _c2 = _sync_sqlite.connect(str(DB_PATH))
                _c2.execute("ALTER TABLE sessions ADD COLUMN role VARCHAR(20) DEFAULT 'default'")
                _c2.commit()
                _c2.close()
            # Migration: add role_name column to messages if missing
            _c2 = _sync_sqlite.connect(str(DB_PATH))
            _msg_cols = [r[1] for r in _c2.execute("PRAGMA table_info('messages')").fetchall()]
            _c2.close()
            if "role_name" not in _msg_cols:
                logger.info("Adding 'role_name' column to messages table")
                _c3 = _sync_sqlite.connect(str(DB_PATH))
                _c3.execute("ALTER TABLE messages ADD COLUMN role_name VARCHAR(20) DEFAULT 'default'")
                _c3.commit()
                _c3.close()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ---------------------------------------------------------------------------
# In-memory chain cache (key = "session_id:role_name")
# ---------------------------------------------------------------------------
_sessions: dict[str, ConversationBufferMemory] = {}


def _memory_key(session_id: str, role_name: str) -> str:
    return f"{session_id}:{role_name}"


# ---------------------------------------------------------------------------
# Database helpers (async)
# ---------------------------------------------------------------------------


async def _get_user(username: str) -> User | None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()


async def _get_or_create_user(username: str) -> User:
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
        if not user:
            user = User(username=username)
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user


async def _get_sessions_for_user(user_id: int) -> list[dict]:
    async with AsyncSessionLocal() as db:
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


async def _get_session(session_id: str) -> dict | None:
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(SessionModel).where(SessionModel.id == session_id))).scalar_one_or_none()
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


async def _verify_ownership(session_id: str, user_id: int) -> dict:
    """Raise 404/403 if session doesn't exist or belongs to another user."""
    s = await _get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    if s["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return s


async def _create_session(
    session_id: str, user_id: int, title: str = "", role: str = "default"
) -> None:
    async with AsyncSessionLocal() as db:
        db.add(SessionModel(id=session_id, title=title, user_id=user_id, role=role))
        await db.commit()


async def _delete_session(session_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(delete(SessionModel).where(SessionModel.id == session_id))
        await db.commit()
    # Clear all role-specific memories for this session
    for role in ROLE_PROMPTS:
        _sessions.pop(_memory_key(session_id, role), None)


async def _rename_session(session_id: str, title: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(title=title, updated_at=func.now())
        )
        await db.commit()


async def _update_session_role(session_id: str, role: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(role=role, updated_at=func.now())
        )
        await db.commit()


async def _get_messages_for_role(session_id: str, role_name: str) -> list[dict]:
    async with AsyncSessionLocal() as db:
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


async def _save_message(
    session_id: str, role: str, content: str, role_name: str = "default"
) -> None:
    async with AsyncSessionLocal() as db:
        db.add(
            MessageModel(
                session_id=session_id, role=role, content=content, role_name=role_name
            )
        )
        await db.execute(
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(updated_at=func.now())
        )
        await db.commit()


async def _auto_title(session_id: str, message: str) -> None:
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(SessionModel).where(SessionModel.id == session_id))).scalar_one_or_none()
        if row and not row.title:
            row.title = message[:20]
            await db.commit()


def _get_memory(session_id: str, role_name: str) -> ConversationBufferMemory:
    """Return existing or new memory for (session, role) pair."""
    key = _memory_key(session_id, role_name)
    if key not in _sessions:
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        _sessions[key] = memory
    return _sessions[key]


async def _load_memory(session_id: str, role_name: str) -> ConversationBufferMemory:
    """Load role-specific messages from DB into memory."""
    memory = _get_memory(session_id, role_name)
    memory.chat_memory.clear()
    messages = await _get_messages_for_role(session_id, role_name)
    for msg in messages:
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            memory.chat_memory.add_ai_message(msg["content"])
    return memory


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _build_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    name = model_name or DEFAULT_MODEL
    cfg = MODEL_REGISTRY.get(name)
    if not cfg:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return ChatOpenAI(
        model=name,
        openai_api_key=cfg["api_key"],
        openai_api_base=cfg["base_url"],
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: str
    model: Optional[str] = None
    user_id: int
    role: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class LoginRequest(BaseModel):
    username: str


class SessionCreateRequest(BaseModel):
    user_id: int
    role: str = "default"


class RenameRequest(BaseModel):
    title: str
    user_id: int


class RoleUpdateRequest(BaseModel):
    role: str
    user_id: int


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Chatbot API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    await _init_db()


# -- Health ---------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# -- Model List -----------------------------------------------------------


@app.get("/models")
async def list_models():
    return {"models": list(MODEL_REGISTRY.keys()), "default": DEFAULT_MODEL}


# -- User Login -----------------------------------------------------------


@app.post("/users/login")
async def login(req: LoginRequest):
    """Get or create a user by username. Returns user info."""
    if not req.username.strip():
        raise HTTPException(status_code=400, detail="Username required")
    user = await _get_or_create_user(req.username.strip())
    return {
        "id": user.id,
        "username": user.username,
        "created_at": str(user.created_at),
    }


# -- Chat (non-streaming) -------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        await _verify_ownership(req.session_id, req.user_id)
        await _load_memory(req.session_id)
        chain = ConversationChain(
            llm=_build_llm(req.model),
            memory=_get_memory(req.session_id),
            verbose=False,
        )
        await _save_message(req.session_id, "user", req.message)
        result = await chain.ainvoke({"input": req.message})
        response_text = result.get("response", "")
        await _save_message(req.session_id, "assistant", response_text)
        await _auto_title(req.session_id, req.message)
        return ChatResponse(response=response_text, session_id=req.session_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail="Internal server error")


# -- Chat (SSE streaming) -------------------------------------------------


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Send a message and stream the AI response via Server-Sent Events."""

    async def event_stream():
        try:
            await _verify_ownership(req.session_id, req.user_id)
            session = await _get_session(req.session_id)
            role_name = req.role or "default"
            system_prompt = ROLE_PROMPTS.get(role_name, ROLE_PROMPTS["default"])

            # Persist role to session record
            if session and session.get("role") != role_name:
                await _update_session_role(req.session_id, role_name)

            llm = _build_llm(req.model)
            memory = await _load_memory(req.session_id, role_name)

            messages = [SystemMessage(content=system_prompt)]
            messages.extend(memory.chat_memory.messages)
            messages.append(HumanMessage(content=req.message))

            await _save_message(req.session_id, "user", req.message, role_name)

            full: str = ""
            async for chunk in llm.astream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    full += content
                    yield (
                        "data: "
                        + json.dumps(
                            {"content": content, "session_id": req.session_id},
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

            memory.save_context({"input": req.message}, {"response": full})
            await _save_message(req.session_id, "assistant", full, role_name)
            await _auto_title(req.session_id, req.message)

            yield (
                "data: "
                + json.dumps({"done": True, "session_id": req.session_id})
                + "\n\n"
            )

        except HTTPException:
            yield f"data: {json.dumps({'error': 'Access denied'})}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client")
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# -- Session CRUD -----------------------------------------------------------


@app.get("/sessions")
async def list_sessions(user_id: int = Query(...)):
    """Return sessions for a user, ordered by most recently updated."""
    return {"sessions": await _get_sessions_for_user(user_id)}


@app.post("/sessions")
async def create_session(req: SessionCreateRequest):
    """Create a new empty session for a user."""
    sid = uuid.uuid4().hex[:8]
    await _create_session(sid, req.user_id)
    return await _get_session(sid)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: int = Query(...)):
    """Delete a session (must be owned by user)."""
    await _verify_ownership(session_id, user_id)
    await _delete_session(session_id)
    return {"ok": True}


@app.put("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    """Rename a session (must be owned by user)."""
    await _verify_ownership(session_id, req.user_id)
    await _rename_session(session_id, req.title)
    return await _get_session(session_id)


@app.put("/sessions/{session_id}/role")
async def update_session_role(session_id: str, req: RoleUpdateRequest):
    """Update session role and return that role's message history."""
    await _verify_ownership(session_id, req.user_id)
    if req.role not in ROLE_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Unknown role: {req.role}")
    await _update_session_role(session_id, req.role)
    messages = await _get_messages_for_role(session_id, req.role)
    return {
        "session": await _get_session(session_id),
        "messages": messages,
    }


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str, user_id: int = Query(...), role_name: str = Query("default")
):
    """Return role-specific messages for a session (must be owned by user)."""
    await _verify_ownership(session_id, user_id)
    return {"messages": await _get_messages_for_role(session_id, role_name)}
