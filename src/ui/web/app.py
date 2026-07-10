"""
Chatbot 前端 — Gradio Web UI
提供流式聊天、多用户支持、角色/人设切换和会话管理功能
"""

import asyncio
import json
import logging
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
):
    """主聊天处理器 — 逐 token 流式输出 AI 回复，完成后显示 token 用量"""
    if not message.strip():
        yield history, gr.update(), session_id, gr.update()
        return

    if not user_id:
        history.append({"role": "assistant", "content": "⚠️ 请先登录后再发送消息"})
        yield history, gr.update(), session_id, gr.update()
        return

    if not session_id:
        session = await _create_session_api(user_id, role, model)
        if not session:
            history.append({"role": "assistant", "content": "⚠️ 无法创建会话"})
            yield history, gr.update(), "", gr.update()
            return
        session_id = session["id"]

    history.append({"role": "user", "content": message})
    yield history, gr.update(value=""), session_id, gr.update()

    history.append({"role": "assistant", "content": ""})

    try:
        usage_out: dict = {}
        async for token in _stream_tokens(message, session_id, model, user_id, role, usage_out):
            history[-1]["content"] += token
            yield history, gr.update(), session_id, gr.update()

        # 流结束，获取累计 token 统计
        if usage_out:
            cumulative = await _fetch_session_tokens(session_id, user_id)
            token_md = _format_tokens(usage_out, cumulative)
            yield history, gr.update(), session_id, token_md
        else:
            yield history, gr.update(), session_id, gr.update()

    except httpx.ConnectError:
        history[-1]["content"] = (
            "⚠️ 无法连接到后端服务，请确认服务已启动。"
        )
        yield history, gr.update(), session_id, gr.update()
    except httpx.ReadTimeout:
        history[-1]["content"] += "\n\n⚠️ 请求超时，请重试。"
        yield history, gr.update(), session_id, gr.update()
    except RuntimeError as e:
        history[-1]["content"] = f"⚠️ {e}"
        yield history, gr.update(), session_id, gr.update()
    except Exception as e:
        history[-1]["content"] = f"⚠️ 发生未知错误：{e}"
        yield history, gr.update(), session_id, gr.update()


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
    if not message.strip():
        yield (*empty, gr.update())
        return
    if not user_id:
        yield (*empty, gr.update())
        return
    if len(models) < 2:
        yield (*empty, gr.update())
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

    # 初始渲染
    yield (*_render(), gr.update(interactive=False))

    try:
        async for event in _parallel_stream_tokens(message, session_id, model_names, user_id, role):
            if "all_done" in event:
                break
            model = event.get("model", "")
            if model not in model_names:
                continue
            if "content" in event:
                buffers[model] += event["content"]
                yield (*_render(), gr.update(interactive=False))
            elif event.get("done"):
                done_models.add(model)
                if "duration" in event:
                    durations[model] = str(event["duration"])
                if "error" in event:
                    errors[model] = event["error"]
                yield (*_render(), gr.update(interactive=False))

    except Exception as e:
        for m in model_names:
            if m not in done_models and m not in errors:
                errors[m] = str(e)
                done_models.add(m)
        yield (*_render(), gr.update(interactive=True))
        return

    yield (*_render(), gr.update(interactive=True))


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


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面"""
    with gr.Blocks(title="Chatbot — LangChain Chat v1.1") as demo:

        # --- 状态 ---
        session_state = gr.State("")
        user_state = gr.State(None)
        username_state = gr.State("")
        role_state = gr.State("default")
        delete_pending_state = gr.State(False)
        session_delete_pending_state = gr.State(False)
        preset_edit_mode = gr.State("none")      # "none" | "create" | "edit"
        preset_edit_id = gr.State(None)
        preset_delete_pending = gr.State(False)

        # --- 页头 ---
        gr.Markdown("# 🤖 Chatbot")
        status_md = gr.Markdown("⏳ 正在检查后端...")
        user_md = gr.Markdown("### ⚠️ 请先登录")

        # --- 登录行 ---
        with gr.Row(equal_height=True):
            login_box = gr.Textbox(
                placeholder="输入用户名...",
                label="用户名",
                scale=4,
                container=False,
            )
            login_btn = gr.Button("登录 / 切换", scale=1, variant="primary", size="sm")
            delete_user_btn = gr.Button("🗑 删除用户", scale=1, variant="stop", size="sm")

        # --- 控制行 ---
        with gr.Row(equal_height=True):
            model_dd = gr.Dropdown(
                label="模型",
                choices=["deepseek-chat"],
                value="deepseek-chat",
                scale=2,
                interactive=True,
            )
            role_dd = gr.Dropdown(
                label="🎭 角色",
                choices=ROLE_OPTIONS,
                value="default",
                scale=2,
                interactive=True,
            )
            session_dd = gr.Dropdown(
                label="会话",
                choices=[],
                value=None,
                scale=3,
                interactive=True,
            )
            new_btn = gr.Button("＋ 新会话", scale=1, variant="secondary", size="sm")
            export_btn = gr.Button("📥 导出", scale=1, variant="secondary", size="sm")
            del_btn = gr.Button("🗑", scale=0, variant="stop", size="sm", min_width=40)
            cancel_del_btn = gr.Button("取消", scale=0, variant="secondary", size="sm", visible=False)

        # --- 重命名行 ---
        with gr.Row(equal_height=True):
            rename_box = gr.Textbox(
                placeholder="输入新标题...",
                label="重命名",
                scale=4,
                container=False,
            )
            rename_btn = gr.Button("✏️ 重命名", scale=1, variant="secondary", size="sm")

        # --- 导出状态 ---
        export_status_md = gr.Markdown("")

        # --- 并行多模型对比 ---
        with gr.Accordion("🔀 多模型并行对比", open=False):
            with gr.Row(equal_height=True):
                parallel_models_cbg = gr.CheckboxGroup(
                    label="选择模型（至少 2 个）",
                    choices=["deepseek-chat"],
                    value=[],
                    scale=4,
                )
                parallel_send_btn = gr.Button("🚀 并行发送", scale=1, variant="primary", size="sm")
            parallel_msg_box = gr.Textbox(
                placeholder="输入消息，Enter 发送到所有选中模型...",
                label="并行提问",
                container=False,
            )
            # 4 个模型输出区域（2x2 网格）
            with gr.Row(equal_height=False):
                parallel_md1 = gr.Markdown("", visible=False)
                parallel_md2 = gr.Markdown("", visible=False)
            with gr.Row(equal_height=False):
                parallel_md3 = gr.Markdown("", visible=False)
                parallel_md4 = gr.Markdown("", visible=False)

        # --- 预设管理按钮 ---
        with gr.Row(equal_height=True):
            preset_mgmt_btn = gr.Button("⚙️ 预设管理", scale=1, variant="secondary", size="sm")

        # --- 预设管理区域（默认隐藏）---
        with gr.Group(visible=False) as preset_mgmt_group:
            gr.Markdown("### 🎭 预设管理")
            with gr.Row(equal_height=True):
                preset_list_dd = gr.Dropdown(
                    label="我的预设",
                    choices=[],
                    value=None,
                    scale=4,
                    interactive=True,
                )
                preset_add_btn = gr.Button("➕ 新增", scale=1, variant="primary", size="sm")
                preset_edit_btn = gr.Button("✏️ 编辑", scale=1, variant="secondary", size="sm")
                preset_del_btn = gr.Button("🗑 删除", scale=1, variant="stop", size="sm")

            # 删除确认按钮
            preset_del_confirm_btn = gr.Button("⚠️ 确认删除", variant="stop", size="sm", visible=False)

            # 新增/编辑表单
            with gr.Group(visible=False) as preset_form_group:
                preset_name_box = gr.Textbox(label="预设名称", placeholder="输入预设名称...")
                preset_desc_box = gr.Textbox(label="描述", placeholder="简短描述...")
                preset_prompt_box = gr.Textbox(
                    label="System Prompt",
                    placeholder="输入系统提示词...",
                    lines=4,
                )
                with gr.Row(equal_height=True):
                    preset_save_btn = gr.Button("💾 保存", variant="primary", scale=1)
                    preset_cancel_btn = gr.Button("❌ 取消", variant="secondary", scale=1)

        # --- 聊天显示区 ---
        chatbot = gr.Chatbot(
            height=520,
            avatar_images=(USER_AVATAR, BOT_AVATAR),
            label="",
        )

        # --- Token 用量显示 ---
        token_md = gr.Markdown("")

        # --- 输入行 ---
        with gr.Row(equal_height=True):
            msg_box = gr.Textbox(
                placeholder="输入消息，Enter 发送...",
                label="",
                scale=10,
                container=False,
            )
            send_btn = gr.Button("发送", scale=1, variant="primary")

        # --- 快捷提问 ---
        gr.Examples(
            examples=["你好，介绍一下自己", "用 Python 写一个快速排序", "推荐三本科幻小说"],
            inputs=[msg_box],
            label="快捷提问",
        )

        # --- 搜索 ---
        with gr.Accordion("🔍 对话搜索", open=False):
            with gr.Row(equal_height=True):
                search_box = gr.Textbox(
                    placeholder="输入关键词搜索历史消息...",
                    label="关键词",
                    scale=4,
                    container=False,
                )
                search_btn = gr.Button("搜索", scale=1, variant="primary", size="sm")
            search_results_md = gr.Markdown("")

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
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state],
            outputs=[chatbot, msg_box, session_state, token_md],
        )

        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state],
            outputs=[chatbot, msg_box, session_state, token_md],
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
            outputs=[parallel_md1, parallel_md2, parallel_md3, parallel_md4, parallel_send_btn],
        )

        parallel_msg_box.submit(
            parallel_chat_respond,
            inputs=[parallel_msg_box, parallel_models_cbg, session_state, user_state, role_state],
            outputs=[parallel_md1, parallel_md2, parallel_md3, parallel_md4, parallel_send_btn],
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
        css="""
            footer { display: none !important; }
            #status-row { align-items: center; }
        """,
    )
