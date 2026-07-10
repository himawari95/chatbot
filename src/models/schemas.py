"""
Pydantic 数据模型定义
包含请求/响应模型、业务实体模型和角色预设模型
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# 请求模型（API 入参）
# =============================================================================


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: str
    model: Optional[str] = None
    user_id: int
    role: str = "default"


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str


class SessionCreateRequest(BaseModel):
    """创建会话请求"""
    user_id: int
    role: str = "default"
    model_name: str = "deepseek-chat"


class RenameRequest(BaseModel):
    """重命名会话请求"""
    title: str
    user_id: int


class RoleUpdateRequest(BaseModel):
    """更新会话角色请求"""
    role: str
    user_id: int


class PresetCreateRequest(BaseModel):
    """创建预设请求"""
    user_id: int
    name: str
    description: str = ""
    system_prompt: str


class PresetUpdateRequest(BaseModel):
    """更新预设请求"""
    user_id: int
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None


# =============================================================================
# 响应模型（API 出参）
# =============================================================================


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str


class ModelsResponse(BaseModel):
    """模型列表响应"""
    models: list[str]
    default: str


# =============================================================================
# 业务实体模型
# =============================================================================


class UserInfo(BaseModel):
    """用户信息"""
    id: int
    username: str
    created_at: str


class SessionInfo(BaseModel):
    """会话信息"""
    id: str
    title: str
    role: str = "default"
    user_id: Optional[int] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    created_at: str = ""
    updated_at: str = ""


class MessageInfo(BaseModel):
    """消息信息"""
    role: str          # "user" 或 "assistant"
    content: str


class PresetInfo(BaseModel):
    """角色预设"""
    name: str          # 内部标识，如 "teacher"
    label: str         # 显示名称，如 "老师"
    emoji: str         # 表情符号
    system_prompt: str # 系统提示词


class UserConfig(BaseModel):
    """用户配置"""
    default_model: str = "deepseek-chat"
    temperature: float = 0.7


# =============================================================================
# 会话角色切换响应
# =============================================================================


class RoleSwitchResponse(BaseModel):
    """角色切换响应"""
    session: SessionInfo
    messages: list[MessageInfo]


# =============================================================================
# 会话列表响应
# =============================================================================


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: list[SessionInfo]


class MessageListResponse(BaseModel):
    """消息列表响应"""
    messages: list[MessageInfo]


class TokenInfo(BaseModel):
    """Token 用量信息"""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class SearchResultItem(BaseModel):
    """搜索结果项"""
    session_id: str
    session_title: str
    content: str
    timestamp: str


class SearchResponse(BaseModel):
    """搜索响应"""
    results: list[SearchResultItem]
