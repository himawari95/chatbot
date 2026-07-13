"""
Chatbot 前端 — Gradio Web UI
提供流式聊天、多用户支持、角色/人设切换和会话管理功能
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

import gradio as gr
import httpx

from src.core.preset_manager import get_preset_manager
from src.interface.ui_protocol import BackendConfig, BOT_AVATAR, USER_AVATAR

# =============================================================================
# 常量
# =============================================================================

_backend = BackendConfig()
BACKEND_URL = _backend.base_url

logger = logging.getLogger("chatbot")

# 加载前端语音模块 JavaScript
_VOICE_JS_PATH = Path(__file__).parent / "static" / "voice.js"
_VOICE_JS_CONTENT = ""
if _VOICE_JS_PATH.exists():
    _VOICE_JS_CONTENT = _VOICE_JS_PATH.read_text(encoding="utf-8")

# 从预设管理器加载角色选项
_presets = get_preset_manager()
ROLE_OPTIONS = _presets.to_choices()


# =============================================================================
# API 辅助函数（异步）
# =============================================================================


async def _health_check() -> str:
    """检测后端服务是否可用"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND_URL}/health", timeout=5.0)
            if r.status_code == 200:
                return "### 🟢 后端已连接"
    except Exception:
        pass
    return "### 🔴 后端不可用 — 请先启动服务"


async def _fetch_models() -> list[str]:
    """获取可用模型列表"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND_URL}/models", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("models", ["deepseek-chat"])
    except Exception:
        pass
    return ["deepseek-chat"]


async def _login_user(username: str) -> dict | None:
    """调用后端登录接口"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{BACKEND_URL}/users/login",
                json={"username": username},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _fetch_sessions(user_id: int) -> list[dict]:
    """获取用户的会话列表"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{BACKEND_URL}/sessions", params={"user_id": user_id}, timeout=5.0
            )
            if r.status_code == 200:
                return r.json().get("sessions", [])
    except Exception:
        pass
    return []


async def _create_session_api(user_id: int, role: str = "default", model_name: str = "deepseek-chat") -> dict | None:
    """调用后端创建新会话"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{BACKEND_URL}/sessions",
                json={"user_id": user_id, "role": role, "model_name": model_name},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _delete_session_api(session_id: str, user_id: int) -> bool:
    """调用后端删除会话"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.delete(
                f"{BACKEND_URL}/sessions/{session_id}",
                params={"user_id": user_id},
                timeout=5.0,
            )
            return r.status_code == 200
    except Exception:
        return False


async def _rename_session_api(session_id: str, title: str, user_id: int) -> dict | None:
    """调用后端重命名会话"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.put(
                f"{BACKEND_URL}/sessions/{session_id}",
                json={"title": title, "user_id": user_id},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _update_role_api(session_id: str, role: str, user_id: int) -> dict | None:
    """调用后端更新会话角色"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.put(
                f"{BACKEND_URL}/sessions/{session_id}/role",
                json={"role": role, "user_id": user_id},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _fetch_messages(
    session_id: str, user_id: int, role_name: str = "default"
) -> list[dict]:
    """获取指定会话和角色的消息历史"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{BACKEND_URL}/sessions/{session_id}/messages",
                params={"user_id": user_id, "role_name": role_name},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json().get("messages", [])
    except Exception:
        pass
    return []


async def _delete_user_api(user_id: int) -> bool:
    """调用后端删除用户"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.delete(
                f"{BACKEND_URL}/users/{user_id}",
                timeout=5.0,
            )
            return r.status_code == 200
    except Exception:
        return False


async def _fetch_presets_api(user_id: int) -> list[dict]:
    """获取所有可用预设（内置 + 用户自定义）"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{BACKEND_URL}/presets",
                params={"user_id": user_id},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json().get("presets", [])
    except Exception:
        pass
    return []


async def _create_preset_api(user_id: int, name: str, description: str, system_prompt: str) -> dict | None:
    """调用后端创建自定义预设"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{BACKEND_URL}/presets",
                json={"user_id": user_id, "name": name, "description": description, "system_prompt": system_prompt},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _update_preset_api(preset_id: int, user_id: int, name: str | None = None, description: str | None = None, system_prompt: str | None = None) -> dict | None:
    """调用后端更新自定义预设"""
    try:
        async with httpx.AsyncClient() as c:
            body = {"user_id": user_id}
            if name is not None:
                body["name"] = name
            if description is not None:
                body["description"] = description
            if system_prompt is not None:
                body["system_prompt"] = system_prompt
            r = await c.put(
                f"{BACKEND_URL}/presets/{preset_id}",
                json=body,
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _delete_preset_api(preset_id: int, user_id: int) -> bool:
    """调用后端删除自定义预设"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.delete(
                f"{BACKEND_URL}/presets/{preset_id}",
                params={"user_id": user_id},
                timeout=5.0,
            )
            return r.status_code == 200
    except Exception:
        return False


async def _fetch_session_tokens(session_id: str, user_id: int) -> dict:
    """获取会话累计 token 用量"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{BACKEND_URL}/sessions/{session_id}/tokens",
                params={"user_id": user_id},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0}


async def _parallel_stream_tokens(
    message: str, session_id: str, models: list[str], user_id: int, role: str,
) -> AsyncGenerator[dict, None]:
    """连接后端并行 SSE 端点，逐模型产出事件"""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as c:
        async with c.stream(
            "POST",
            f"{BACKEND_URL}/chat/parallel",
            json={
                "message": message,
                "session_id": session_id,
                "models": models,
                "user_id": user_id,
                "role": role,
            },
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise RuntimeError(f"后端返回 {resp.status_code}: {text[:200]}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if "error" in payload and "all_done" not in payload and "model" not in payload:
                    raise RuntimeError(payload["error"])
                yield payload


async def _upload_file_api(
    file_data: bytes, filename: str, user_id: int, session_id: str,
    message: str = "", role: str = "default",
) -> dict | None:
    """调用后端文件上传与多模态对话接口"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as c:
            r = await c.post(
                f"{BACKEND_URL}/chat/upload",
                params={
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": role,
                    "message": message,
                },
                files={"file": (filename, file_data)},
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _export_session_api(session_id: str, user_id: int) -> dict | None:
    """调用后端导出会话"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{BACKEND_URL}/sessions/{session_id}/export",
                params={"user_id": user_id},
                timeout=10.0,
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


async def _search_messages_api(user_id: int, keyword: str) -> list[dict]:
    """搜索用户消息"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{BACKEND_URL}/search",
                params={"user_id": user_id, "keyword": keyword},
                timeout=5.0,
            )
            if r.status_code == 200:
                return r.json().get("results", [])
    except Exception:
        pass
    return []


def _format_tokens(last_usage: dict, cumulative: dict) -> str:
    """格式化 token 用量为 Markdown"""
    lp = last_usage.get("prompt_tokens", 0)
    lc = last_usage.get("completion_tokens", 0)
    lt = last_usage.get("total_tokens", 0)
    cp = cumulative.get("total_prompt_tokens", 0)
    cc = cumulative.get("total_completion_tokens", 0)
    ct = cumulative.get("total_tokens", 0)
    lines = []
    if lt > 0:
        lines.append(f"**本轮**：输入 {lp} | 输出 {lc} | 合计 {lt}")
    if ct > 0:
        lines.append(f"**累计**：输入 {cp} | 输出 {cc} | 合计 {ct}")
    return "  \n".join(lines) if lines else ""


async def _refresh_role_choices(user_id: int) -> list[tuple[str, str]]:
    """刷新角色下拉选项（内置 + 用户自定义）"""
    if not user_id:
        return ROLE_OPTIONS
    presets = await _fetch_presets_api(user_id)
    choices = []
    for p in presets:
        emoji = p.get("emoji", "🤖")
        label = p.get("label", p["name"])
        choices.append((f"{emoji} {label}", p["name"]))
    return choices


async def _build_dropdown(user_id: int) -> tuple[list, str | None]:
    """构建会话下拉选项（格式：标题 | 模型 | 预设 | 时间）"""
    sessions = await _fetch_sessions(user_id)
    choices = []
    for s in sessions:
        title = s["title"] if s["title"] else s["id"]
        model = s.get("model_name", "deepseek-chat")
        role_key = s.get("role", "default")
        role_label = _presets.get_preset(role_key).label
        created = s.get("created_at", "")[:16].replace("T", " ")
        label = f"{title} | {model} | {role_label} | {created}"
        choices.append((label, s["id"]))
    default_id = sessions[0]["id"] if sessions else None
    return choices, default_id


# =============================================================================
# 流式辅助
# =============================================================================


async def _stream_tokens(
    message: str, session_id: str, model: str, user_id: int, role: str,
    usage_out: dict | None = None,
) -> AsyncGenerator[str, None]:
    """连接后端 SSE 端点，逐 token 产出内容。
    若传入 usage_out，则在流结束后将 token 用量写入该字典。
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as c:
        async with c.stream(
            "POST",
            f"{BACKEND_URL}/chat/stream",
            json={
                "message": message,
                "session_id": session_id,
                "model": model,
                "user_id": user_id,
                "role": role,
            },
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                raise RuntimeError(f"后端返回 {resp.status_code}: {text[:200]}")

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "error" in payload:
                    raise RuntimeError(payload["error"])

                if "content" in payload:
                    yield payload["content"]

                if "done" in payload:
                    if usage_out is not None and "usage" in payload:
                        usage_out.update(payload["usage"])
                    return


# =============================================================================
# UI 事件处理器
# =============================================================================


async def login(username: str) -> tuple:
    """用户登录或切换用户，加载会话列表和历史"""
    if not username.strip():
        return (
            gr.update(),   # user_state
            "",            # username_state
            "### ⚠️ 请输入用户名",
            gr.update(),   # session_dd
            [],            # chatbot
            "",            # session_state
            gr.update(),   # msg_box
            gr.update(value=""),  # rename_box
            "default",     # role_state
            False,         # delete_pending_state
            gr.update(value="🗑 删除用户", variant="stop"),  # delete_user_btn
            False,         # session_delete_pending_state
            gr.update(value="🗑", variant="stop"),  # del_btn
            gr.update(visible=False),  # cancel_del_btn
            gr.update(),   # role_dd
        )

    user = await _login_user(username.strip())
    if not user:
        return (
            gr.update(),
            "",
            "### 🔴 登录失败 — 请检查后端是否正常",
            gr.update(),
            [], "", gr.update(), gr.update(value=""),
            "default",
            False,
            gr.update(value="🗑 删除用户", variant="stop"),
            False,
            gr.update(value="🗑", variant="stop"),
            gr.update(visible=False),
            gr.update(),
        )

    user_id = user["id"]
    username_val = user["username"]
    logger.info("UI 用户登录", extra={"user_id": user_id, "username": username_val, "operation": "ui_login"})
    choices, default_id = await _build_dropdown(user_id)
    role_choices = await _refresh_role_choices(user_id)

    if default_id:
        sessions = await _fetch_sessions(user_id)
        session_role = "default"
        for s in sessions:
            if s["id"] == default_id:
                session_role = s.get("role", "default")
                break
        messages = await _fetch_messages(default_id, user_id, session_role)
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
    else:
        history = []
        session_role = "default"

    return (
        user_id,
        username_val,
        f"### 🟢 当前用户：**{username_val}**",
        gr.update(choices=choices, value=default_id),
        history,
        default_id or "",
        gr.update(),
        gr.update(value=""),
        session_role,
        False,
        gr.update(value="🗑 删除用户", variant="stop"),
        False,
        gr.update(value="🗑", variant="stop"),
        gr.update(visible=False),
        gr.update(choices=role_choices, value=session_role),
    )


async def delete_user(
    user_id: int, username: str, delete_pending: bool
) -> tuple:
    """删除用户 — 首次点击进入确认状态，再次点击执行删除"""
    if not user_id:
        return (
            gr.update(), "", "### ⚠️ 请先登录后再操作",
            gr.update(), [], "", gr.update(),
            gr.update(value=""), "default", False,
            gr.update(value="🗑 删除用户", variant="stop"),
            False, gr.update(value="🗑", variant="stop"), gr.update(visible=False),
            gr.update(),
        )

    if not delete_pending:
        # 第一次点击：进入确认状态
        return (
            gr.update(), username,
            f"### ⚠️ 确定要删除用户 **{username}** 吗？此操作将删除该用户的所有数据（会话、消息），不可恢复！",
            gr.update(), [], "", gr.update(),
            gr.update(value=""), "default", True,
            gr.update(value="⚠️ 确认删除", variant="stop"),
            False, gr.update(value="🗑", variant="stop"), gr.update(visible=False),
            gr.update(),
        )

    # 第二次点击：执行删除
    success = await _delete_user_api(user_id)
    logger.info("UI 用户删除", extra={"user_id": user_id, "operation": "ui_delete_user", "success": success})
    if success:
        return (
            None, "", "### ⚠️ 用户已删除，请重新登录",
            gr.update(choices=[], value=None), [], "", gr.update(),
            gr.update(value=""), "default", False,
            gr.update(value="🗑 删除用户", variant="stop"),
            False, gr.update(value="🗑", variant="stop"), gr.update(visible=False),
            gr.update(),
        )
    else:
        return (
            user_id, username,
            "### 🔴 删除失败 — 请检查后端是否正常",
            gr.update(), [], "", gr.update(),
            gr.update(value=""), "default", False,
            gr.update(value="🗑 删除用户", variant="stop"),
            False, gr.update(value="🗑", variant="stop"), gr.update(visible=False),
            gr.update(),
        )


async def respond(
    message: str,
    history: list[dict],
    session_id: str,
    model: str,
    user_id: int,
    role: str,
    file_path: str | None = None,
):
    """聊天处理器 — 支持纯文本和文件上传"""
    if not message.strip() and not file_path:
        yield history, gr.update(), session_id, gr.update(), gr.update()
        return

    if not user_id:
        history.append({"role": "assistant", "content": "⚠️ 请先登录后再发送消息"})
        yield history, gr.update(), session_id, gr.update(), gr.update()
        return

    if not session_id:
        session = await _create_session_api(user_id, role, model)
        if not session:
            history.append({"role": "assistant", "content": "⚠️ 无法创建会话"})
            yield history, gr.update(), "", gr.update(), gr.update()
            return
        session_id = session["id"]

    # 文件上传分支
    if file_path:
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
        except Exception as e:
            history.append({"role": "assistant", "content": f"⚠️ 读取文件失败：{e}"})
            yield history, gr.update(), session_id, gr.update(), gr.update(value=None)
            return

        filename = file_path.replace("\\", "/").split("/")[-1]
        display = f"[文件：{filename}] {message}" if message.strip() else f"[文件：{filename}]"
        history.append({"role": "user", "content": display})
        history.append({"role": "assistant", "content": "⏳ 正在处理文件..."})
        yield history, gr.update(value=""), session_id, gr.update(), gr.update(value=None)

        result = await _upload_file_api(file_data, filename, user_id, session_id, message, role)
        if not result:
            history[-1]["content"] = "⚠️ 文件处理失败"
            yield history, gr.update(), session_id, gr.update(), gr.update(value=None)
            return
        history[-1]["content"] = result["response"]
        yield history, gr.update(), session_id, gr.update(), gr.update(value=None)
        return

    # 纯文本分支（原有逻辑）
    history.append({"role": "user", "content": message})
    yield history, gr.update(value=""), session_id, gr.update(), gr.update()

    history.append({"role": "assistant", "content": ""})

    try:
        usage_out: dict = {}
        async for token in _stream_tokens(message, session_id, model, user_id, role, usage_out):
            history[-1]["content"] += token
            yield history, gr.update(), session_id, gr.update(), gr.update()

        # 流结束，获取累计 token 统计
        if usage_out:
            cumulative = await _fetch_session_tokens(session_id, user_id)
            token_md = _format_tokens(usage_out, cumulative)
            yield history, gr.update(), session_id, token_md, gr.update()
        else:
            yield history, gr.update(), session_id, gr.update(), gr.update()

    except httpx.ConnectError:
        history[-1]["content"] = "⚠️ 无法连接到后端服务，请确认服务已启动。"
        yield history, gr.update(), session_id, gr.update(), gr.update()
    except httpx.ReadTimeout:
        history[-1]["content"] += "\n\n⚠️ 请求超时，请重试。"
        yield history, gr.update(), session_id, gr.update(), gr.update()
    except RuntimeError as e:
        history[-1]["content"] = f"⚠️ {e}"
        yield history, gr.update(), session_id, gr.update(), gr.update()
    except Exception as e:
        history[-1]["content"] = f"⚠️ 发生未知错误：{e}"
        yield history, gr.update(), session_id, gr.update(), gr.update()


async def create_session(user_id: int, role: str, model_name: str) -> tuple:
    """创建新会话"""
    if not user_id:
        return "", [], gr.update(), False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)
    session = await _create_session_api(user_id, role, model_name)
    if not session:
        return "", [], gr.update(), False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)
    new_id = session["id"]
    choices, _ = await _build_dropdown(user_id)
    return new_id, [], gr.update(choices=choices, value=new_id), False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)


async def switch_session(session_id: str, user_id: int) -> tuple:
    """切换到指定会话，加载对应角色的消息历史"""
    if not session_id or not user_id:
        return [], "", gr.update(), "default", False, gr.update(value="🗑", variant="stop"), gr.update(visible=False), ""

    sessions = await _fetch_sessions(user_id)
    session_role = "default"
    for s in sessions:
        if s["id"] == session_id:
            session_role = s.get("role", "default")
            break

    messages = await _fetch_messages(session_id, user_id, session_role)
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    choices, _ = await _build_dropdown(user_id)
    tokens = await _fetch_session_tokens(session_id, user_id)
    token_md = _format_tokens({}, tokens)

    return history, session_id, gr.update(choices=choices, value=session_id), session_role, False, gr.update(value="🗑", variant="stop"), gr.update(visible=False), token_md


async def delete_session(
    session_id: str, user_id: int, delete_pending: bool
) -> tuple[list, str, dict, str, bool, dict, dict]:
    """删除当前会话 — 首次点击进入确认状态，显示取消按钮，再次点击确认执行删除"""
    if not session_id or not user_id:
        return gr.update(), "", gr.update(), "default", False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)

    if not delete_pending:
        # 第一次点击：进入确认状态，显示取消按钮（不修改聊天窗口内容）
        return gr.update(), session_id, gr.update(), "default", True, gr.update(value="⚠️ 确认删除", variant="stop"), gr.update(visible=True)

    # 第二次点击：执行删除
    await _delete_session_api(session_id, user_id)

    choices, default_id = await _build_dropdown(user_id)
    if default_id:
        sessions = await _fetch_sessions(user_id)
        session_role = "default"
        for s in sessions:
            if s["id"] == default_id:
                session_role = s.get("role", "default")
                break
        messages = await _fetch_messages(default_id, user_id, session_role)
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
    else:
        history = []
        session_role = "default"
    return history, default_id or "", gr.update(choices=choices, value=default_id), session_role, False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)


async def cancel_session_delete() -> tuple[bool, dict, dict]:
    """取消会话删除确认"""
    return False, gr.update(value="🗑", variant="stop"), gr.update(visible=False)


async def rename_session(
    session_id: str, new_title: str, user_id: int
) -> tuple[dict, dict]:
    """重命名当前会话"""
    if not session_id or not new_title.strip() or not user_id:
        choices, _ = await _build_dropdown(user_id) if user_id else ([], None)
        return gr.update(choices=choices, value=session_id), gr.update()
    await _rename_session_api(session_id, new_title.strip(), user_id)
    choices, _ = await _build_dropdown(user_id)
    return gr.update(choices=choices, value=session_id), gr.update(value="")


async def change_role(
    session_id: str, role: str, user_id: int
) -> tuple[list, str, str, dict]:
    """切换角色：加载该角色对应的消息历史"""
    if not user_id:
        return [], session_id, role, gr.update()
    if session_id:
        result = await _update_role_api(session_id, role, user_id)
        if result and "messages" in result:
            messages = result["messages"]
            history = [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
            return history, session_id, role, gr.update()
    return [], session_id, role, gr.update()


# =============================================================================
# 预设管理事件处理器
# =============================================================================


async def open_preset_manager(user_id: int):
    """打开预设管理界面，加载预设列表"""
    if not user_id:
        return (
            gr.update(choices=[], value=None),       # preset_list_dd
            gr.update(visible=True),                   # preset_mgmt_group
            gr.update(visible=False),                  # preset_form_group
            gr.update(visible=False),                  # preset_del_confirm_btn
            gr.update(value=""),                       # preset_name_box
            gr.update(value=""),                       # preset_desc_box
            gr.update(value=""),                       # preset_prompt_box
            "none",                                    # preset_edit_mode
            None,                                      # preset_edit_id
            False,                                     # preset_delete_pending
            gr.update(visible=True),                   # preset_add_btn
            gr.update(visible=True),                   # preset_edit_btn
            gr.update(visible=True),                   # preset_del_btn
        )
    presets = await _fetch_presets_api(user_id)
    custom_presets = [p for p in presets if not p.get("is_builtin")]
    choices = [(f"{p['name']} (自定义)", p["id"]) for p in custom_presets]
    return (
        gr.update(choices=choices, value=None),   # preset_list_dd
        gr.update(visible=True),                   # preset_mgmt_group
        gr.update(visible=False),                  # preset_form_group
        gr.update(visible=False),                  # preset_del_confirm_btn
        gr.update(value=""),                       # preset_name_box
        gr.update(value=""),                       # preset_desc_box
        gr.update(value=""),                       # preset_prompt_box
        "none",                                    # preset_edit_mode
        None,                                      # preset_edit_id
        False,                                     # preset_delete_pending
        gr.update(visible=True),                   # preset_add_btn
        gr.update(visible=True),                   # preset_edit_btn
        gr.update(visible=True),                   # preset_del_btn
    )


async def start_create_preset():
    """显示创建预设表单"""
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        "create",
        gr.update(visible=False),
    )


async def start_edit_preset(preset_id: int | None, user_id: int):
    """加载预设数据到编辑表单"""
    if not preset_id or not user_id:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            "none",
            None,
            gr.update(visible=False),
        )
    presets = await _fetch_presets_api(user_id)
    target = None
    for p in presets:
        if p.get("id") == preset_id and not p.get("is_builtin"):
            target = p
            break
    if not target:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            "none",
            None,
            gr.update(visible=False),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=target["name"]),
        gr.update(value=target.get("description", "")),
        gr.update(value=target["system_prompt"]),
        "edit",
        preset_id,
        gr.update(visible=False),
    )


async def save_preset(
    mode: str, preset_id: int | None, user_id: int,
    p_name: str, p_desc: str, p_prompt: str,
):
    """保存预设（创建或更新）
    返回: preset_list_dd, role_dd, preset_form_group, preset_edit_btn,
          preset_name_box, preset_desc_box, preset_prompt_box,
          preset_edit_mode, preset_edit_id, preset_add_btn,
          session_dd, preset_del_btn
    """
    if not user_id or not p_name.strip() or not p_prompt.strip():
        return (
            gr.update(), gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            "none", None,
            gr.update(visible=True), gr.update(),
            gr.update(visible=False),
        )

    if mode == "create":
        await _create_preset_api(user_id, p_name.strip(), p_desc.strip(), p_prompt.strip())
    elif mode == "edit" and preset_id:
        await _update_preset_api(preset_id, user_id, p_name.strip(), p_desc.strip(), p_prompt.strip())
    else:
        return (
            gr.update(), gr.update(),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            "none", None,
            gr.update(visible=True), gr.update(),
            gr.update(visible=False),
        )

    # 刷新预设列表和角色下拉
    presets = await _fetch_presets_api(user_id)
    custom_presets = [p for p in presets if not p.get("is_builtin")]
    p_choices = [(f"{p['name']} (自定义)", p["id"]) for p in custom_presets]
    role_choices = await _refresh_role_choices(user_id)

    return (
        gr.update(choices=p_choices, value=None),  # preset_list_dd
        gr.update(choices=role_choices),            # role_dd
        gr.update(visible=False),                   # preset_form_group
        gr.update(visible=False),                   # preset_edit_btn
        gr.update(value=""),                        # preset_name_box
        gr.update(value=""),                        # preset_desc_box
        gr.update(value=""),                        # preset_prompt_box
        "none",                                     # preset_edit_mode
        None,                                       # preset_edit_id
        gr.update(visible=True),                    # preset_add_btn
        gr.update(),                                # session_dd
        gr.update(visible=False),                   # preset_del_btn
    )


async def delete_preset_confirm(preset_id: int | None):
    """进入删除确认状态"""
    if not preset_id:
        return False, gr.update(visible=False)
    return True, gr.update(visible=True)


async def delete_preset_execute(preset_id: int | None, user_id: int):
    """执行删除预设
    返回: preset_list_dd, role_dd, session_dd,
          preset_delete_pending, preset_del_confirm_btn,
          preset_form_group, preset_edit_btn,
          preset_name_box, preset_desc_box, preset_prompt_box,
          preset_edit_mode, preset_edit_id, preset_add_btn
    """
    if not preset_id or not user_id:
        return (
            gr.update(), gr.update(), gr.update(),
            False, gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            "none", None,
            gr.update(visible=True),
        )

    await _delete_preset_api(preset_id, user_id)

    # 刷新
    presets = await _fetch_presets_api(user_id)
    custom_presets = [p for p in presets if not p.get("is_builtin")]
    p_choices = [(f"{p['name']} (自定义)", p["id"]) for p in custom_presets]
    role_choices = await _refresh_role_choices(user_id)

    return (
        gr.update(choices=p_choices, value=None),  # preset_list_dd
        gr.update(choices=role_choices),            # role_dd
        gr.update(),                                # session_dd
        False,                                      # preset_delete_pending
        gr.update(visible=False),                   # preset_del_confirm_btn
        gr.update(visible=False),                   # preset_form_group
        gr.update(visible=False),                   # preset_edit_btn
        gr.update(value=""),                        # preset_name_box
        gr.update(value=""),                        # preset_desc_box
        gr.update(value=""),                        # preset_prompt_box
        "none",                                     # preset_edit_mode
        None,                                       # preset_edit_id
        gr.update(visible=True),                    # preset_add_btn
    )


async def cancel_preset_form():
    """取消预设编辑/创建"""
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        "none",
        None,
        gr.update(visible=True),
        False,
        gr.update(visible=False),
    )


async def search_messages(user_id: int, keyword: str) -> str:
    """搜索消息并格式化结果"""
    if not user_id:
        return "### ⚠️ 请先登录后再搜索"
    if not keyword.strip():
        return "### ⚠️ 请输入搜索关键词"
    results = await _search_messages_api(user_id, keyword.strip())
    if not results:
        return f"### 🔍 未找到包含 \"{keyword.strip()}\" 的消息"
    lines = [f"### 🔍 搜索结果（{len(results)} 条）：keyword \"{keyword.strip()}\"\n"]
    for i, r in enumerate(results, 1):
        title = r.get("session_title", "") or r["session_id"]
        content = r["content"][:200]
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        lines.append(f"**{i}.** [{title}] {ts}")
        lines.append(f"> {content}")
        lines.append("")
    return "\n".join(lines)


async def upload_and_chat(
    file_path: str | None, message: str, history: list[dict],
    session_id: str, user_id: int, role: str,
):
    """文件上传处理器 — 读取文件并发送到后端进行多模态对话"""
    if not user_id:
        history.append({"role": "assistant", "content": "⚠️ 请先登录"})
        yield history, file_path, gr.update()
        return

    if not file_path:
        yield history, file_path, gr.update()
        return

    if not session_id:
        session = await _create_session_api(user_id, role, "deepseek-chat")
        if not session:
            history.append({"role": "assistant", "content": "⚠️ 无法创建会话"})
            yield history, file_path, gr.update()
            return
        session_id = session["id"]

    # 读取文件
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠️ 读取文件失败：{e}"})
        yield history, file_path, gr.update()
        return

    filename = file_path.replace("\\", "/").split("/")[-1]
    history.append({"role": "user", "content": f"[文件：{filename}] {message}" if message else f"[文件：{filename}]"})
    history.append({"role": "assistant", "content": "⏳ 正在处理文件..."})
    yield history, file_path, gr.update()

    result = await _upload_file_api(file_data, filename, user_id, session_id, message, role)
    if not result:
        history[-1]["content"] = "⚠️ 文件处理失败 — 请检查后端是否正常"
        yield history, file_path, gr.update()
        return

    history[-1]["content"] = result["response"]
    yield history, file_path, gr.update()


async def export_current_session(session_id: str, user_id: int) -> str:
    """导出当前会话到桌面"""
    if not user_id:
        return "### ⚠️ 请先登录后再导出"
    if not session_id:
        return "### ⚠️ 请先选择或创建会话"
    result = await _export_session_api(session_id, user_id)
    if not result:
        logger.warning("UI 导出失败", extra={"user_id": user_id, "session_id": session_id, "operation": "ui_export"})
        return "### 🔴 导出失败 — 请检查后端是否正常"
    filepath = result.get("filepath", "")
    logger.info("UI 导出成功", extra={"user_id": user_id, "session_id": session_id, "operation": "ui_export", "filepath": filepath})
    return f"### ✅ 已导出到桌面 chatbot_exports 文件夹\n> {filepath}"


async def parallel_chat_respond(
    message: str, models: list[str], session_id: str, user_id: int, role: str,
):
    """并行多模型聊天处理器 — 逐模型流式输出"""
    empty = (gr.update(visible=False, value=""),) * 4
    show_placeholder = gr.update(visible=True)
    hide_placeholder = gr.update(visible=False)
    if not message.strip():
        yield (*empty, gr.update(), show_placeholder)
        return
    if not user_id:
        yield (*empty, gr.update(), show_placeholder)
        return
    if len(models) < 2:
        yield (*empty, gr.update(), show_placeholder)
        return
    if not session_id:
        session = await _create_session_api(user_id, role, models[0])
        if not session:
            yield (*empty, gr.update())
            return
        session_id = session["id"]

    # 每个模型对应一个响应缓冲区
    model_names = models[:4]  # 最多 4 个
    buffers: dict[str, str] = {m: "" for m in model_names}
    durations: dict[str, str] = {}
    errors: dict[str, str] = {}
    done_models: set = set()

    def _render() -> tuple:
        """渲染各模型的当前状态为 4 个 Markdown"""
        results = []
        for m in model_names:
            if m in errors:
                results.append(gr.update(
                    visible=True,
                    value=f"### ❌ {m}\n> 错误：{errors[m]}",
                ))
            elif m in done_models:
                d = durations.get(m, "?")
                results.append(gr.update(
                    visible=True,
                    value=f"### ✅ {m}（{d}s）\n{buffers[m]}",
                ))
            else:
                results.append(gr.update(
                    visible=True,
                    value=f"### ⏳ {m}\n{buffers[m]}",
                ))
        # 补齐到 4 个
        while len(results) < 4:
            results.append(gr.update(visible=False, value=""))
        return tuple(results)

    # 初始渲染 — 隐藏占位提示
    yield (*_render(), gr.update(interactive=False), hide_placeholder)

    try:
        async for event in _parallel_stream_tokens(message, session_id, model_names, user_id, role):
            if "all_done" in event:
                break
            model = event.get("model", "")
            if model not in model_names:
                continue
            if "content" in event:
                buffers[model] += event["content"]
                yield (*_render(), gr.update(interactive=False), hide_placeholder)
            elif event.get("done"):
                done_models.add(model)
                if "duration" in event:
                    durations[model] = str(event["duration"])
                if "error" in event:
                    errors[model] = event["error"]
                yield (*_render(), gr.update(interactive=False), hide_placeholder)

    except Exception as e:
        for m in model_names:
            if m not in done_models and m not in errors:
                errors[m] = str(e)
                done_models.add(m)
        yield (*_render(), gr.update(interactive=True), hide_placeholder)
        return

    yield (*_render(), gr.update(interactive=True), hide_placeholder)


def on_startup():
    """页面加载时的初始化回调"""

    async def _init():
        status, models = await asyncio.gather(_health_check(), _fetch_models())
        return (
            status,
            gr.update(choices=models, value=models[0] if models else "deepseek-chat"),
            gr.update(choices=models, value=[]),
        )

    return asyncio.run(_init())


# =============================================================================
# UI 布局
# =============================================================================


# =============================================================================
# 毛玻璃清新风格 CSS（模块级常量，供 build_ui 和 run_ui 共用）
# =============================================================================

GLASS_CSS = """
/* ============================================================
   全局 — 毛玻璃清新风格
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    max-width: 100% !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 50%, #fce4ec 100%) !important;
    min-height: 100vh !important;
}
body {
    background: linear-gradient(135deg, #e8f0fe 0%, #f3e8ff 50%, #fce4ec 100%) !important;
}

/* ---- 主容器 ---- */
.contain, .app, .main-header, .gr-box {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.08) !important;
}

/* ---- 页头 ---- */
h1, .app h1 {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
}

/* ---- 按钮 ---- */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.gr-button-primary:hover, button.primary:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
.gr-button-secondary, button.secondary {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #4a4a6a !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.gr-button-secondary:hover {
    background: rgba(255,255,255,0.7) !important;
    border-color: rgba(99,102,241,0.3) !important;
}
button[class*="stop"], .gr-button-stop {
    background: linear-gradient(135deg, #f43f5e, #e11d48) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ---- 输入框 ---- */
textarea, input[type="text"], .gr-textbox textarea, .gr-textbox input {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    color: #1a1a2e !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}

/* ---- 下拉菜单 ---- */
.gr-dropdown, select, .wrap-inner {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
}
.options, .gr-dropdown .options {
    background: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.12) !important;
}

/* ---- 聊天气泡 ---- */
.chatbot .message, .bubble-wrap {
    border-radius: 20px !important;
    margin: 8px 0 !important;
    animation: fadeInUp 0.3s ease !important;
}
.chatbot .user, .user .bubble-wrap, .message.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}
.chatbot .bot, .bot .bubble-wrap, .message.bot {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    color: #1a1a2e !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 14px 18px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}
.chatbot { background: transparent !important; border: none !important; }
.message-wrap { background: transparent !important; }

/* ---- Accordion ---- */
.gr-accordion, .accordion {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 16px !important;
}

/* ---- 文件上传 ---- */
.gr-file, .file-preview {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(12px) !important;
    border: 2px dashed rgba(99,102,241,0.25) !important;
    border-radius: 16px !important;
}
.gr-file:hover { border-color: rgba(99,102,241,0.5) !important; }

/* ---- 滚动条 ---- */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }

/* ---- 动画 ---- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
}

/* ---- 底部栏 ---- */
footer, .footer {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-top: 1px solid rgba(255,255,255,0.3) !important;
}

/* ---- 标签 ---- */
label, .gr-label { color: #4a4a6a !important; font-weight: 500 !important; font-size: 13px !important; }

/* ---- 面板 ---- */
.gr-panel { background: transparent !important; border: none !important; }
.gr-row, .gr-column { gap: 8px !important; }

/* ---- 隐藏页脚 ---- */
footer.gr-footer { display: none !important; }

/* ---- 语音录音动画 ---- */
#voice-mic-btn[style*="background: rgb(239, 68, 68)"] { animation: pulse 1.5s infinite !important; }

/* ---- 上传按钮 ---- */
.gr-upload-button { border-radius: 12px !important; min-height: 40px !important; }
.gr-upload-button:hover { background: rgba(255,255,255,0.7) !important; border-color: rgba(99,102,241,0.4) !important; }

/* ---- 多模型对比网格卡片 ---- */
.parallel-compare {
    background: rgba(255,255,255,0.35) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}
.parallel-card {
    min-height: 300px !important;
    max-height: 400px !important;
    overflow-y: auto !important;
    background: rgba(255,255,255,0.5) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ---- 多模型对比面板 ---- */
.gr-accordion { margin-top: 8px !important; }

/* ---- 两栏布局响应式 ---- */
@media (max-width: 768px) {
    .gradio-container .gr-row > .gr-column:nth-child(1) { display: none !important; }
}
"""


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面（清新毛玻璃风格 / 两栏布局 + 顶部工具栏）"""
    with gr.Blocks(title="Chatbot — Glass Edition", theme=gr.themes.Soft(), css=GLASS_CSS) as demo:

        # ================================================================
        # 状态变量
        # ================================================================
        session_state = gr.State("")
        user_state = gr.State(None)
        username_state = gr.State("")
        role_state = gr.State("default")
        delete_pending_state = gr.State(False)
        session_delete_pending_state = gr.State(False)
        preset_edit_mode = gr.State("none")
        preset_edit_id = gr.State(None)
        preset_delete_pending = gr.State(False)

        # ================================================================
        # 状态指示
        # ================================================================
        status_md = gr.Markdown("⏳ 正在检查后端...", visible=False)
        user_md = gr.Markdown("### ⚠️ 请先登录", visible=False)

        # ================================================================
        # 语音模块注入（隐藏）
        # ================================================================
        voice_mic_html = gr.HTML(
            value="",
            visible=False,
            head=f"""<script>
{_VOICE_JS_CONTENT}
(function() {{
    function bindMic() {{
        var btn = document.getElementById('voice-mic-btn');
        if (!btn) {{ setTimeout(bindMic, 200); return; }}
        btn.addEventListener('click', function() {{ window.toggleVoiceInput(); }});
    }}
    bindMic();
}})();
(function() {{
    var apiBase = '{BACKEND_URL}';
    var observer = new MutationObserver(function() {{
        document.querySelectorAll('.bot').forEach(function(bot) {{
            if (bot.querySelector('.voice-play-btn')) return;
            var btn = document.createElement('button');
            btn.className = 'voice-play-btn';
            btn.textContent = '🔊'; btn.title = '播放语音';
            btn.style.cssText = 'margin:2px 0;padding:1px 6px;font-size:11px;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5;float:right;';
            btn.onclick = function() {{
                var text = bot.textContent.replace('🔊','').trim();
                if (text) window.playTTS(text, apiBase);
            }};
            bot.appendChild(btn);
        }});
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>""")

        # ================================================================
        # 登录卡片
        # ================================================================
        gr.Markdown("<div style='text-align:center;margin-top:60px'><h1>🤖 Chatbot</h1></div>")
        with gr.Column(scale=1, min_width=360):
            with gr.Group():
                login_box = gr.Textbox(
                    placeholder="输入用户名...",
                    label="用户名",
                    container=True,
                )
                with gr.Row():
                    login_btn = gr.Button("登录 / 切换", variant="primary")
                    delete_user_btn = gr.Button("🗑 删除用户", variant="stop")

        # ================================================================
        # 顶部工具栏（全宽，一行）
        # ================================================================
        with gr.Row():
            search_box = gr.Textbox(
                placeholder="🔍 搜索历史消息...",
                label="",
                scale=10,
            )
            search_btn = gr.Button("🔍", variant="secondary", scale=1, min_width=48)
            model_dd = gr.Dropdown(
                choices=["deepseek-chat"],
                value="deepseek-chat",
                label="模型",
                scale=3,
            )
            role_dd = gr.Dropdown(
                label="🎭 角色",
                choices=ROLE_OPTIONS,
                value="default",
                scale=3,
            )
            export_btn = gr.Button("📥 导出", variant="secondary", scale=2, min_width=80)
            new_btn = gr.Button("＋ 新建会话", variant="primary", scale=2, min_width=100)

        search_results_md = gr.Markdown("")
        export_status_md = gr.Markdown("")

        # ================================================================
        # 主界面（两栏布局）
        # ================================================================
        with gr.Row(equal_height=False):

            # ========== 左侧面板 (1) ==========
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### 👤 用户")

                gr.Markdown("### 💬 会话")
                session_dd = gr.Dropdown(
                    label="",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                with gr.Row():
                    del_btn = gr.Button("🗑", variant="stop", size="sm", min_width=40)
                    cancel_del_btn = gr.Button("取消", variant="secondary", size="sm", visible=False)
                with gr.Row():
                    rename_box = gr.Textbox(placeholder="重命名...", label="", container=False, scale=4)
                    rename_btn = gr.Button("✏️", variant="secondary", size="sm", scale=0, min_width=40)

                preset_mgmt_btn = gr.Button("⚙️ 预设管理", variant="secondary", size="sm")
                with gr.Group(visible=False) as preset_mgmt_group:
                    preset_list_dd = gr.Dropdown(label="我的预设", choices=[], value=None)
                    with gr.Row():
                        preset_add_btn = gr.Button("➕ 新增", variant="primary", size="sm")
                        preset_edit_btn = gr.Button("✏️ 编辑", variant="secondary", size="sm")
                        preset_del_btn = gr.Button("🗑 删除", variant="stop", size="sm")
                    preset_del_confirm_btn = gr.Button("⚠️ 确认删除", variant="stop", size="sm", visible=False)
                    with gr.Group(visible=False) as preset_form_group:
                        preset_name_box = gr.Textbox(label="预设名称")
                        preset_desc_box = gr.Textbox(label="描述")
                        preset_prompt_box = gr.Textbox(label="System Prompt", lines=4)
                        with gr.Row():
                            preset_save_btn = gr.Button("💾 保存", variant="primary")
                            preset_cancel_btn = gr.Button("❌ 取消", variant="secondary")

            # ========== 中间主区域 (4) ==========
            with gr.Column(scale=4, min_width=400):
                chatbot = gr.Chatbot(
                    height=520,
                    avatar_images=(USER_AVATAR, BOT_AVATAR),
                    label="",
                )

                token_md = gr.Markdown("")

                # 多模型并行对比（中下区域板块）
                with gr.Group(elem_classes="parallel-compare"):
                    gr.Markdown("### 🔀 多模型并行对比")
                    with gr.Row():
                        parallel_models_cbg = gr.CheckboxGroup(
                            label="选择模型",
                            choices=["deepseek-chat"],
                            value=[],
                            scale=3,
                        )
                        parallel_msg_box = gr.Textbox(
                            placeholder="输入消息，发送到所有选中模型...",
                            label="并行提问",
                            scale=5,
                            container=False,
                        )
                        parallel_send_btn = gr.Button("🚀 发送", variant="primary", scale=1, min_width=80)
                    parallel_placeholder = gr.Markdown(
                        "> 💡 请选择至少 2 个模型进行对比",
                        visible=True,
                    )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            parallel_md1 = gr.Markdown("", visible=False, elem_classes="parallel-card")
                        with gr.Column(scale=1, min_width=200):
                            parallel_md2 = gr.Markdown("", visible=False, elem_classes="parallel-card")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            parallel_md3 = gr.Markdown("", visible=False, elem_classes="parallel-card")
                        with gr.Column(scale=1, min_width=200):
                            parallel_md4 = gr.Markdown("", visible=False, elem_classes="parallel-card")

                with gr.Row():
                    file_upload = gr.UploadButton(
                        "📎",
                        file_count="single",
                        file_types=[".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".pdf", ".docx", ".txt"],
                        scale=0,
                        min_width=48,
                    )
                    voice_btn_html = gr.HTML(
                        value="""<button id="voice-mic-btn" style="
                            width:100%;height:100%;font-size:18px;cursor:pointer;
                            border:1px solid rgba(0,0,0,0.1);border-radius:8px;
                            background:rgba(255,255,255,0.5);color:#374151;
                            transition:all 0.2s;min-height:40px;
                        " onmouseover="this.style.background='rgba(255,255,255,0.7)'" onmouseout="this.style.background='rgba(255,255,255,0.5)'">
                            🎤
                        </button>""",
                        scale=0,
                        min_width=52,
                    )
                    msg_box = gr.Textbox(
                        placeholder="输入消息，Enter 发送...",
                        label="",
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Examples(
                    examples=["你好，介绍一下自己", "用 Python 写一个快速排序", "推荐三本科幻小说"],
                    inputs=[msg_box],
                    label="快捷提问",
                )

        # ================================================================
        # 事件绑定
        # ================================================================

        login_btn.click(
            login,
            inputs=[login_box],
            outputs=[
                user_state, username_state, user_md, session_dd, chatbot,
                session_state, msg_box, rename_box, role_state,
                delete_pending_state, delete_user_btn,
                session_delete_pending_state, del_btn, cancel_del_btn,
                role_dd,
            ],
        )

        delete_user_btn.click(
            delete_user,
            inputs=[user_state, username_state, delete_pending_state],
            outputs=[
                user_state, username_state, user_md, session_dd, chatbot,
                session_state, msg_box, rename_box, role_state,
                delete_pending_state, delete_user_btn,
                session_delete_pending_state, del_btn, cancel_del_btn,
                role_dd,
            ],
        )

        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state, file_upload],
            outputs=[chatbot, msg_box, session_state, token_md, file_upload],
        )

        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state, file_upload],
            outputs=[chatbot, msg_box, session_state, token_md, file_upload],
        )

        new_btn.click(
            create_session,
            inputs=[user_state, role_state, model_dd],
            outputs=[session_state, chatbot, session_dd,
                     session_delete_pending_state, del_btn, cancel_del_btn],
        )

        session_dd.change(
            switch_session,
            inputs=[session_dd, user_state],
            outputs=[chatbot, session_state, session_dd, role_state,
                     session_delete_pending_state, del_btn, cancel_del_btn,
                     token_md],
        )

        del_btn.click(
            delete_session,
            inputs=[session_state, user_state, session_delete_pending_state],
            outputs=[chatbot, session_state, session_dd, role_state,
                     session_delete_pending_state, del_btn, cancel_del_btn],
        )

        cancel_del_btn.click(
            cancel_session_delete,
            inputs=[],
            outputs=[session_delete_pending_state, del_btn, cancel_del_btn],
        )

        rename_btn.click(
            rename_session,
            inputs=[session_state, rename_box, user_state],
            outputs=[session_dd, rename_box],
        )

        role_dd.change(
            change_role,
            inputs=[session_state, role_dd, user_state],
            outputs=[chatbot, session_state, role_state, session_dd],
        )

        # --- 预设管理事件 ---

        preset_mgmt_btn.click(
            open_preset_manager,
            inputs=[user_state],
            outputs=[
                preset_list_dd, preset_mgmt_group, preset_form_group,
                preset_del_confirm_btn, preset_name_box, preset_desc_box,
                preset_prompt_box, preset_edit_mode, preset_edit_id,
                preset_delete_pending, preset_add_btn, preset_edit_btn,
                preset_del_btn,
            ],
        )

        preset_add_btn.click(
            start_create_preset,
            inputs=[],
            outputs=[
                preset_form_group, preset_edit_btn, preset_del_btn,
                preset_name_box, preset_desc_box, preset_prompt_box,
                preset_edit_mode, preset_add_btn,
            ],
        )

        preset_edit_btn.click(
            start_edit_preset,
            inputs=[preset_list_dd, user_state],
            outputs=[
                preset_form_group, preset_edit_btn, preset_name_box,
                preset_desc_box, preset_prompt_box, preset_edit_mode,
                preset_edit_id, preset_add_btn,
            ],
        )

        preset_save_btn.click(
            save_preset,
            inputs=[
                preset_edit_mode, preset_edit_id, user_state,
                preset_name_box, preset_desc_box, preset_prompt_box,
            ],
            outputs=[
                preset_list_dd, role_dd, preset_form_group,
                preset_edit_btn, preset_name_box, preset_desc_box,
                preset_prompt_box, preset_edit_mode, preset_edit_id,
                preset_add_btn, session_dd, preset_del_btn,
            ],
        )

        preset_del_btn.click(
            delete_preset_confirm,
            inputs=[preset_list_dd],
            outputs=[preset_delete_pending, preset_del_confirm_btn],
        )

        preset_del_confirm_btn.click(
            delete_preset_execute,
            inputs=[preset_list_dd, user_state],
            outputs=[
                preset_list_dd, role_dd, session_dd,
                preset_delete_pending, preset_del_confirm_btn,
                preset_form_group, preset_edit_btn,
                preset_name_box, preset_desc_box, preset_prompt_box,
                preset_edit_mode, preset_edit_id, preset_add_btn,
            ],
        )

        preset_cancel_btn.click(
            cancel_preset_form,
            inputs=[],
            outputs=[
                preset_form_group, preset_edit_btn,
                preset_name_box, preset_desc_box, preset_prompt_box,
                preset_edit_mode, preset_edit_id, preset_add_btn,
                preset_delete_pending, preset_del_confirm_btn,
            ],
        )

        # --- 导出事件 ---

        export_btn.click(
            export_current_session,
            inputs=[session_state, user_state],
            outputs=[export_status_md],
        )

        # --- 并行聊天事件 ---

        parallel_send_btn.click(
            parallel_chat_respond,
            inputs=[parallel_msg_box, parallel_models_cbg, session_state, user_state, role_state],
            outputs=[parallel_md1, parallel_md2, parallel_md3, parallel_md4, parallel_send_btn, parallel_placeholder],
        )

        parallel_msg_box.submit(
            parallel_chat_respond,
            inputs=[parallel_msg_box, parallel_models_cbg, session_state, user_state, role_state],
            outputs=[parallel_md1, parallel_md2, parallel_md3, parallel_md4, parallel_send_btn, parallel_placeholder],
        )

        # --- 搜索事件 ---

        search_btn.click(
            search_messages,
            inputs=[user_state, search_box],
            outputs=[search_results_md],
        )

        demo.load(
            on_startup,
            outputs=[status_md, model_dd, parallel_models_cbg],
        )

    return demo


# =============================================================================
# 入口
# =============================================================================

_demo = build_ui()

if __name__ == "__main__":
    _demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=GLASS_CSS,
    )
