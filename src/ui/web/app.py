"""
Chatbot 前端 — Gradio Web UI
提供流式聊天、多用户支持、角色/人设切换和会话管理功能
"""

import asyncio
import json
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


async def _create_session_api(user_id: int, role: str = "default") -> dict | None:
    """调用后端创建新会话"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{BACKEND_URL}/sessions",
                json={"user_id": user_id, "role": role},
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


async def _build_dropdown(user_id: int) -> tuple[list, str | None]:
    """构建会话下拉选项"""
    sessions = await _fetch_sessions(user_id)
    choices = []
    for s in sessions:
        label = s["title"] if s["title"] else s["id"]
        choices.append((label, s["id"]))
    default_id = sessions[0]["id"] if sessions else None
    return choices, default_id


# =============================================================================
# 流式辅助
# =============================================================================


async def _stream_tokens(
    message: str, session_id: str, model: str, user_id: int, role: str
) -> AsyncGenerator[str, None]:
    """连接后端 SSE 端点，逐 token 产出内容"""
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
                    return


# =============================================================================
# UI 事件处理器
# =============================================================================


async def login(username: str) -> tuple:
    """用户登录或切换用户，加载会话列表和历史"""
    if not username.strip():
        return (
            gr.update(),
            "### ⚠️ 请输入用户名",
            gr.update(),
            [], "", gr.update(), gr.update(value=""),
            "default",
        )

    user = await _login_user(username.strip())
    if not user:
        return (
            gr.update(),
            "### 🔴 登录失败 — 请检查后端是否正常",
            gr.update(),
            [], "", gr.update(), gr.update(value=""),
            "default",
        )

    user_id = user["id"]
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

    return (
        user_id,
        f"### 🟢 当前用户：**{user['username']}**",
        gr.update(choices=choices, value=default_id),
        history,
        default_id or "",
        gr.update(),
        gr.update(value=""),
        session_role,
    )


async def respond(
    message: str,
    history: list[dict],
    session_id: str,
    model: str,
    user_id: int,
    role: str,
):
    """主聊天处理器 — 逐 token 流式输出 AI 回复"""
    if not message.strip():
        yield history, gr.update(), session_id
        return

    if not user_id:
        history.append({"role": "assistant", "content": "⚠️ 请先登录后再发送消息"})
        yield history, gr.update(), session_id
        return

    if not session_id:
        session = await _create_session_api(user_id, role)
        if not session:
            history.append({"role": "assistant", "content": "⚠️ 无法创建会话"})
            yield history, gr.update(), ""
            return
        session_id = session["id"]

    history.append({"role": "user", "content": message})
    yield history, gr.update(value=""), session_id

    history.append({"role": "assistant", "content": ""})

    try:
        async for token in _stream_tokens(message, session_id, model, user_id, role):
            history[-1]["content"] += token
            yield history, gr.update(), session_id

    except httpx.ConnectError:
        history[-1]["content"] = (
            "⚠️ 无法连接到后端服务，请确认服务已启动。"
        )
        yield history, gr.update(), session_id
    except httpx.ReadTimeout:
        history[-1]["content"] += "\n\n⚠️ 请求超时，请重试。"
        yield history, gr.update(), session_id
    except RuntimeError as e:
        history[-1]["content"] = f"⚠️ {e}"
        yield history, gr.update(), session_id
    except Exception as e:
        history[-1]["content"] = f"⚠️ 发生未知错误：{e}"
        yield history, gr.update(), session_id


async def create_session(user_id: int, role: str) -> tuple[str, list, dict]:
    """创建新会话"""
    if not user_id:
        return "", [], gr.update()
    session = await _create_session_api(user_id, role)
    if not session:
        return "", [], gr.update()
    new_id = session["id"]
    choices, _ = await _build_dropdown(user_id)
    return new_id, [], gr.update(choices=choices, value=new_id)


async def switch_session(session_id: str, user_id: int) -> tuple[list, str, dict, str]:
    """切换到指定会话，加载对应角色的消息历史"""
    if not session_id or not user_id:
        return [], "", gr.update(), "default"

    sessions = await _fetch_sessions(user_id)
    session_role = "default"
    for s in sessions:
        if s["id"] == session_id:
            session_role = s.get("role", "default")
            break

    messages = await _fetch_messages(session_id, user_id, session_role)
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    choices, _ = await _build_dropdown(user_id)

    return history, session_id, gr.update(choices=choices, value=session_id), session_role


async def delete_session(session_id: str, user_id: int) -> tuple[list, str, dict, str]:
    """删除当前会话并自动切换到另一个"""
    if not session_id or not user_id:
        return [], "", gr.update(), "default"
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
    return history, default_id or "", gr.update(choices=choices, value=default_id), session_role


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


def on_startup():
    """页面加载时的初始化回调"""

    async def _init():
        status, models = await asyncio.gather(_health_check(), _fetch_models())
        return (
            status,
            gr.update(choices=models, value=models[0] if models else "deepseek-chat"),
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
        role_state = gr.State("default")

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
            del_btn = gr.Button("🗑", scale=0, variant="stop", size="sm", min_width=40)

        # --- 重命名行 ---
        with gr.Row(equal_height=True):
            rename_box = gr.Textbox(
                placeholder="输入新标题...",
                label="重命名",
                scale=4,
                container=False,
            )
            rename_btn = gr.Button("✏️ 重命名", scale=1, variant="secondary", size="sm")

        # --- 聊天显示区 ---
        chatbot = gr.Chatbot(
            height=520,
            avatar_images=(USER_AVATAR, BOT_AVATAR),
            label="",
        )

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

        # ================================================================
        # 事件绑定
        # ================================================================

        login_btn.click(
            login,
            inputs=[login_box],
            outputs=[
                user_state, user_md, session_dd, chatbot,
                session_state, msg_box, rename_box, role_state,
            ],
        )

        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state],
            outputs=[chatbot, msg_box, session_state],
        )

        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, session_state, model_dd, user_state, role_state],
            outputs=[chatbot, msg_box, session_state],
        )

        new_btn.click(
            create_session,
            inputs=[user_state, role_state],
            outputs=[session_state, chatbot, session_dd],
        )

        session_dd.change(
            switch_session,
            inputs=[session_dd, user_state],
            outputs=[chatbot, session_state, session_dd, role_state],
        )

        del_btn.click(
            delete_session,
            inputs=[session_state, user_state],
            outputs=[chatbot, session_state, session_dd, role_state],
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

        demo.load(
            on_startup,
            outputs=[status_md, model_dd],
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
