"""
UI 协议定义
定义前端 UI 与后端 API 之间的通信协议和类型约束
"""

from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# 后端 API 协议
# =============================================================================


@dataclass
class BackendConfig:
    """后端连接配置"""
    base_url: str = "http://127.0.0.1:8000"
    timeout_connect: float = 10.0
    timeout_read: float = 60.0


# =============================================================================
# 角色选项协议
# =============================================================================


@dataclass
class RoleOption:
    """角色下拉选项"""
    label: str       # 显示名称（含 emoji）
    value: str       # 内部标识


# =============================================================================
# UI 状态协议
# =============================================================================


@dataclass
class UIState:
    """UI 全局状态快照"""
    session_id: str = ""
    user_id: Optional[int] = None
    role: str = "default"
    model: str = "deepseek-chat"
    username: str = ""
    sessions: list[dict] = field(default_factory=list)


# =============================================================================
# 消息格式协议（与后端 SSE / REST 对齐）
# =============================================================================


@dataclass
class ChatMessage:
    """聊天消息（Gradio Chatbot 兼容格式）"""
    role: str        # "user" | "assistant"
    content: str


@dataclass
class SSEEvent:
    """SSE 流式事件"""
    content: Optional[str] = None    # 文本增量
    done: bool = False               # 流是否结束
    session_id: str = ""             # 会话 ID
    error: Optional[str] = None      # 错误信息


# =============================================================================
# 前端常量
# =============================================================================

BOT_AVATAR = "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=chatbot"
USER_AVATAR = None
