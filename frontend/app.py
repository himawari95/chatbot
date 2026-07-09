"""
Chatbot Frontend — Gradio UI
Streaming chat, multi-user support, role/persona switching, session management.
"""

import asyncio
import json
from typing import AsyncGenerator

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"
BOT_AVATAR = "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=chatbot"
USER_AVATAR = None

ROLE_OPTIONS = [
    ("默认", "default"),
    ("👨‍🏫 老师", "teacher"),
    ("👨‍💻 程序员", "programmer"),
    ("🧠 哲学家", "philosopher"),
    ("🤝 朋友", "friend"),
]


# ===================================================================
# API Helpers (async)
# ===================================================================


async def _health_check() -> str:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND_URL}/health", timeout=5.0)
            if r.status_code == 200:
                return "### 🟢 后端已连接"
    except Exception:
        pass
    return "### 🔴 后端不可用 — 请先启动 `uvicorn backend.main:app`"


async def _fetch_models() -> list[str]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{BACKEND_URL}/models", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("models", ["deepseek-chat"])
    except Exception:
        pass
    return ["deepseek-chat"]


async def _login_user(username: str) -> dict | None:
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
    """Build (choices, default_id) for session dropdown."""
    sessions = await _fetch_sessions(user_id)
    choices = []
    for s in sessions:
        label = s["title"] if s["title"] else s["id"]
        choices.append((label, s["id"]))
    default_id = sessions[0]["id"] if sessions else None
    return choices, default_id


# ===================================================================
# Streaming helper
# ===================================================================


async def _stream_tokens(
    message: str, session_id: str, model: str, user_id: int, role: str
) -> AsyncGenerator[str, None]:
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
                raise RuntimeError(f"Backend returned {resp.status_code}: {text[:200]}")

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


# ===================================================================
# UI Event Handlers
# ===================================================================


async def login(username: str) -> tuple:
    """Login or switch user. Clears chat and reloads sessions."""
    if not username.strip():
        return (
            gr.update(),  # user_state
            "### ⚠️ 请输入用户名",
            gr.update(),  # session_dd
            [],           # chatbot
            "",           # session_state
            gr.update(),  # msg_box
            gr.update(value=""),  # rename_box
            "default",    # role_state
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
    """Main chat handler — streams assistant reply token-by-token."""
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
            "⚠️ 无法连接到后端服务，请确认 `uvicorn backend.main:app` 已启动。"
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
    """Create a new session for the current user with the given role."""
    if not user_id:
        return "", [], gr.update()
    session = await _create_session_api(user_id, role)
    if not session:
        return "", [], gr.update()
    new_id = session["id"]
    choices, _ = await _build_dropdown(user_id)
    return new_id, [], gr.update(choices=choices, value=new_id)


async def switch_session(session_id: str, user_id: int) -> tuple[list, str, dict, str]:
    """Load chat history and restore role for the selected session."""
    if not session_id or not user_id:
        return [], "", gr.update(), "default"

    # Restore role from session, then load role-specific messages
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
    """Delete the selected session and switch to another."""
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
    """Rename the current session. Clears rename box on success."""
    if not session_id or not new_title.strip() or not user_id:
        choices, _ = await _build_dropdown(user_id) if user_id else ([], None)
        return gr.update(choices=choices, value=session_id), gr.update()
    await _rename_session_api(session_id, new_title.strip(), user_id)
    choices, _ = await _build_dropdown(user_id)
    return gr.update(choices=choices, value=session_id), gr.update(value="")


async def change_role(
    session_id: str, role: str, user_id: int
) -> tuple[list, str, str, dict]:
    """Switch role: load that role's message history into the chat window."""
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
    """Run once when page loads — health check, fetch models."""

    async def _init():
        status, models = await asyncio.gather(_health_check(), _fetch_models())
        return (
            status,
            gr.update(choices=models, value=models[0] if models else "deepseek-chat"),
        )

    return asyncio.run(_init())


# ===================================================================
# UI Layout
# ===================================================================


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Chatbot") as demo:

        # --- State ---
        session_state = gr.State("")
        user_state = gr.State(None)
        role_state = gr.State("default")

        # --- Header ---
        gr.Markdown("# 🤖 Chatbot")
        status_md = gr.Markdown("⏳ 正在检查后端...")
        user_md = gr.Markdown("### ⚠️ 请先登录")

        # --- Login Row ---
        with gr.Row(equal_height=True):
            login_box = gr.Textbox(
                placeholder="输入用户名...",
                label="用户名",
                scale=4,
                container=False,
            )
            login_btn = gr.Button("登录 / 切换", scale=1, variant="primary", size="sm")

        # --- Controls Row ---
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

        # --- Rename Row ---
        with gr.Row(equal_height=True):
            rename_box = gr.Textbox(
                placeholder="输入新标题...",
                label="重命名",
                scale=4,
                container=False,
            )
            rename_btn = gr.Button("✏️ 重命名", scale=1, variant="secondary", size="sm")

        # --- Chat Display ---
        chatbot = gr.Chatbot(
            height=520,
            avatar_images=(USER_AVATAR, BOT_AVATAR),
            label="",
        )

        # --- Input Row ---
        with gr.Row(equal_height=True):
            msg_box = gr.Textbox(
                placeholder="输入消息，Enter 发送...",
                label="",
                scale=10,
                container=False,
            )
            send_btn = gr.Button("发送", scale=1, variant="primary")

        # --- Example prompts ---
        gr.Examples(
            examples=["你好，介绍一下自己", "用 Python 写一个快速排序", "推荐三本科幻小说"],
            inputs=[msg_box],
            label="快捷提问",
        )

        # ============================================================
        # Event Wiring
        # ============================================================

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


# ===================================================================
# Entry Point
# ===================================================================

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
