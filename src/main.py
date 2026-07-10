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
    ParallelChatRequest,
    PresetCreateRequest,
    PresetUpdateRequest,
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
_chat_engine = ChatEngine(_session_manager, _presets, _config, _storage)


# =============================================================================
# FastAPI 生命周期
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化日志、数据库，关闭时释放资源"""
    _config.init_logging()
    logger.info("应用启动", extra={"operation": "startup"})
    logger.info("正在初始化数据库...")
    await _storage.initialize()
    logger.info("数据库初始化完成")
    yield
    logger.info("正在关闭数据库连接...")
    await _storage.close()
    logger.info("应用关闭", extra={"operation": "shutdown"})


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
# 辅助函数
# =============================================================================


async def _validate_role(role: str, user_id: int) -> bool:
    """验证角色是否有效（内置预设 + 用户自定义预设）"""
    if role in _presets.list_preset_names():
        return True
    preset = await _storage.get_preset_by_name(role, user_id)
    return preset is not None


async def _resolve_system_prompt(role: str, user_id: int) -> str:
    """解析角色的系统提示词（内置优先，其次用户自定义）"""
    if _presets.exists(role):
        return _presets.get_system_prompt(role)
    preset = await _storage.get_preset_by_name(role, user_id)
    if preset:
        return preset["system_prompt"]
    return _presets.get_system_prompt("default")


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
async def list_presets(user_id: int = Query(...)):
    """返回所有可用的角色预设（内置 + 用户自定义）"""
    builtin = [
        {
            "id": None,
            "name": p.name,
            "label": p.label,
            "emoji": p.emoji,
            "description": p.label,
            "system_prompt": p.system_prompt,
            "is_builtin": True,
        }
        for p in _presets.list_presets()
    ]
    user_presets = await _storage.get_presets_for_user(user_id)
    custom = [
        {
            "id": p["id"],
            "name": p["name"],
            "label": p["name"],
            "emoji": "👤",
            "description": p.get("description", ""),
            "system_prompt": p["system_prompt"],
            "is_builtin": False,
        }
        for p in user_presets
    ]
    return {"presets": builtin + custom}


@app.post("/presets")
async def create_preset(req: PresetCreateRequest):
    """创建用户自定义预设"""
    try:
        # 检查名称是否与内置预设冲突
        if req.name in _presets.list_preset_names():
            raise HTTPException(status_code=400, detail="预设名称与内置预设冲突")
        # 检查用户是否已有同名预设
        existing = await _storage.get_preset_by_name(req.name, req.user_id)
        if existing:
            raise HTTPException(status_code=400, detail="已存在同名预设")
        preset = await _storage.create_preset(
            req.user_id, req.name, req.description, req.system_prompt
        )
        return preset
    except HTTPException:
        raise
    except Exception:
        logger.exception("创建预设失败")
        raise HTTPException(status_code=500, detail="创建预设失败")


@app.put("/presets/{preset_id}")
async def update_preset(preset_id: int, req: PresetUpdateRequest):
    """更新用户自定义预设"""
    try:
        preset = await _storage.update_preset(
            preset_id, req.user_id, req.name, req.description, req.system_prompt
        )
        if not preset:
            raise HTTPException(status_code=404, detail="预设不存在或无权修改")
        return preset
    except HTTPException:
        raise
    except Exception:
        logger.exception("更新预设失败")
        raise HTTPException(status_code=500, detail="更新预设失败")


@app.delete("/presets/{preset_id}")
async def delete_preset(preset_id: int, user_id: int = Query(...)):
    """删除用户自定义预设"""
    try:
        ok = await _storage.delete_preset(preset_id, user_id)
        if not ok:
            raise HTTPException(status_code=404, detail="预设不存在或无权删除")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception:
        logger.exception("删除预设失败")
        raise HTTPException(status_code=500, detail="删除预设失败")


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
        logger.exception("删除用户失败", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="删除用户失败")


# --- 非流式聊天 ---


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """发送消息并获取完整 AI 回复（非流式）"""
    try:
        if not await _validate_role(req.role, req.user_id):
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
        logger.exception("聊天请求处理出错", extra={"user_id": req.user_id, "session_id": req.session_id, "model": req.model})
        raise HTTPException(status_code=500, detail="服务器内部错误")


# --- 流式聊天（SSE） ---


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发送消息并以 SSE 格式流式返回 AI 回复"""
    if not await _validate_role(req.role, req.user_id):
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


# --- 并行多模型聊天（SSE） ---


@app.post("/chat/parallel")
async def chat_parallel(req: ParallelChatRequest):
    """并行调用多个模型，以 SSE 格式流式返回对比结果"""
    if len(req.models) < 2:
        raise HTTPException(status_code=400, detail="至少需要选择 2 个模型")
    if not await _validate_role(req.role, req.user_id):
        raise HTTPException(status_code=400, detail=f"未知角色: {req.role}")

    async def event_stream():
        async for event in _chat_engine.parallel_chat_stream(
            message=req.message,
            model_names=req.models,
            session_id=req.session_id,
            user_id=req.user_id,
            role=req.role,
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
        user_id=req.user_id, role=req.role, model_name=req.model_name,
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
        if not await _validate_role(req.role, req.user_id):
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


# --- Token 统计 ---


@app.get("/sessions/{session_id}/tokens")
async def get_session_tokens(session_id: str, user_id: int = Query(...)):
    """获取会话累计 token 用量（需验证所有权）"""
    try:
        await _session_manager.verify_ownership(session_id, user_id)
        tokens = await _session_manager.get_session_tokens(session_id)
        return tokens
    except HTTPException:
        raise
    except Exception:
        logger.exception("获取 token 统计失败")
        raise HTTPException(status_code=500, detail="获取 token 统计失败")


# --- 对话搜索 ---


@app.get("/search")
async def search_messages(user_id: int = Query(...), keyword: str = Query(...)):
    """搜索当前用户所有会话中的消息"""
    try:
        results = await _session_manager.search_messages(user_id, keyword)
        return {"results": results}
    except Exception:
        logger.exception("搜索消息失败")
        raise HTTPException(status_code=500, detail="搜索失败")


# --- 对话导出 ---


@app.post("/sessions/{session_id}/export")
async def export_session(session_id: str, user_id: int = Query(...)):
    """导出会话为 Markdown 文件（需验证所有权）"""
    try:
        filepath = await _session_manager.export_session(session_id, user_id)
        return {"ok": True, "filepath": filepath}
    except HTTPException:
        raise
    except Exception:
        logger.exception("导出会话失败", extra={"user_id": user_id, "session_id": session_id})
        raise HTTPException(status_code=500, detail="导出失败")


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
