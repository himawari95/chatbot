"""
Chatbot 统一入口
启动 FastAPI 后端 + Gradio Web UI，整合所有核心模块
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.core.chat_engine import ChatEngine
from src.core.config_manager import get_config
from src.core.preset_manager import get_preset_manager
from src.core.session_manager import SessionManager
from src.core.user_manager import UserManager
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    LoginRequest,
    RenameRequest,
    RoleUpdateRequest,
    SessionCreateRequest,
)
from src.storage.factory import create_storage_backend

logger = logging.getLogger("chatbot")

# =============================================================================
# 初始化全局组件
# =============================================================================

_config = get_config()
_presets = get_preset_manager()

# 创建存储后端并初始化
_storage = create_storage_backend(
    backend_type=_config.db_type, db_path=_config.db_path
)

# 创建业务管理器
_user_manager = UserManager(_storage)
_session_manager = SessionManager(_storage)
_chat_engine = ChatEngine(_session_manager, _presets, _config)


# =============================================================================
# FastAPI 生命周期
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库，关闭时释放资源"""
    logger.info("正在初始化数据库...")
    await _storage.initialize()
    logger.info("数据库初始化完成")
    yield
    logger.info("正在关闭数据库连接...")
    await _storage.close()
    logger.info("数据库连接已关闭")


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title="Chatbot API",
    version="1.1.0",
    description="LangChain Chat — 多用户、多角色、流式对话 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API 端点
# =============================================================================


# --- 健康检查 ---


@app.get("/health")
async def health():
    """服务健康检查"""
    db_ok = await _storage.health_check()
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


# --- 模型列表 ---


@app.get("/models")
async def list_models():
    """返回可用的 LLM 模型列表"""
    return _chat_engine.list_models()


# --- 角色预设列表 ---


@app.get("/presets")
async def list_presets():
    """返回所有可用的角色预设"""
    presets = _presets.list_presets()
    return {
        "presets": [
            {
                "name": p.name,
                "label": p.label,
                "emoji": p.emoji,
                "system_prompt": p.system_prompt,
            }
            for p in presets
        ]
    }


# --- 用户登录 ---


@app.post("/users/login")
async def login(req: LoginRequest):
    """用户登录（获取或创建用户）"""
    try:
        user = await _user_manager.login(req.username)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- 用户删除 ---


@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """删除用户及其所有关联数据（会话、消息），不可恢复"""
    try:
        await _user_manager.delete_user(user_id)
        return {"ok": True}
    except Exception:
        logger.exception("删除用户失败")
        raise HTTPException(status_code=500, detail="删除用户失败")


# --- 非流式聊天 ---


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """发送消息并获取完整 AI 回复（非流式）"""
    try:
        if req.role not in _presets.list_preset_names():
            raise HTTPException(status_code=400, detail=f"未知角色: {req.role}")
        result = await _chat_engine.chat(
            message=req.message,
            session_id=req.session_id,
            user_id=req.user_id,
            role=req.role,
            model=req.model,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("聊天请求处理出错")
        raise HTTPException(status_code=500, detail="服务器内部错误")


# --- 流式聊天（SSE） ---


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发送消息并以 SSE 格式流式返回 AI 回复"""
    if req.role not in _presets.list_preset_names():
        raise HTTPException(status_code=400, detail=f"未知角色: {req.role}")

    async def event_stream():
        async for event in _chat_engine.chat_stream(
            message=req.message,
            session_id=req.session_id,
            user_id=req.user_id,
            role=req.role,
            model=req.model,
        ):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- 会话 CRUD ---


@app.get("/sessions")
async def list_sessions(user_id: int = Query(...)):
    """获取用户的会话列表"""
    sessions = await _session_manager.list_sessions(user_id)
    return {"sessions": sessions}


@app.post("/sessions")
async def create_session(req: SessionCreateRequest):
    """创建新会话"""
    session = await _session_manager.create_session(
        user_id=req.user_id, role=req.role
    )
    return session


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: int = Query(...)):
    """删除会话（需验证所有权）"""
    try:
        await _session_manager.delete_session(session_id, user_id)
        return {"ok": True}
    except HTTPException:
        raise


@app.put("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    """重命名会话（需验证所有权）"""
    try:
        session = await _session_manager.rename_session(
            session_id, req.title, req.user_id
        )
        return session
    except HTTPException:
        raise


@app.put("/sessions/{session_id}/role")
async def update_session_role(session_id: str, req: RoleUpdateRequest):
    """更新会话角色并返回该角色的消息历史"""
    try:
        if req.role not in _presets.list_preset_names():
            raise HTTPException(status_code=400, detail=f"未知角色: {req.role}")
        await _session_manager.update_role(session_id, req.role, req.user_id)
        messages = await _session_manager.get_messages(session_id, req.role)
        session = await _session_manager.get_session(session_id)
        return {"session": session, "messages": messages}
    except HTTPException:
        raise


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str, user_id: int = Query(...), role_name: str = Query("default")
):
    """获取指定会话和角色的消息历史（需验证所有权）"""
    try:
        await _session_manager.verify_ownership(session_id, user_id)
        messages = await _session_manager.get_messages(session_id, role_name)
        return {"messages": messages}
    except HTTPException:
        raise


# =============================================================================
# 启动入口
# =============================================================================


def run_backend():
    """独立启动 FastAPI 后端"""
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=_config.server_host,
        port=_config.server_port,
        reload=False,
        log_level="info",
    )


def run_ui():
    """独立启动 Gradio 前端"""
    from src.ui.web.app import _demo
    _demo.launch(
        server_name=_config.ui_host,
        server_port=_config.ui_port,
        share=_config.ui_share,
        show_error=True,
        css="""
            footer { display: none !important; }
            #status-row { align-items: center; }
        """,
    )


async def run_all():
    """同时启动后端和前端"""
    import threading
    import uvicorn

    # 先初始化存储
    await _storage.initialize()

    # 在独立线程中启动 Gradio
    ui_thread = threading.Thread(
        target=run_ui,
        daemon=True,
    )
    ui_thread.start()

    # 在主线程中启动 Uvicorn
    config = uvicorn.Config(
        "src.main:app",
        host=_config.server_host,
        port=_config.server_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_all())
