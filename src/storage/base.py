"""
存储层抽象基类
定义所有 CRUD 操作的接口，便于后续替换存储后端（如 PostgreSQL、MongoDB）
"""

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """存储后端抽象基类，所有存储实现必须继承此类"""

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """初始化数据库连接并创建表结构"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """关闭数据库连接"""
        ...

    # ------------------------------------------------------------------
    # 用户操作
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_user(self, username: str) -> dict | None:
        """根据用户名获取用户信息"""
        ...

    @abstractmethod
    async def get_or_create_user(self, username: str) -> dict:
        """获取或创建用户，返回用户信息字典"""
        ...

    @abstractmethod
    async def delete_user(self, user_id: int) -> None:
        """删除用户及其所有关联数据（会话、消息），不可恢复"""
        ...

    # ------------------------------------------------------------------
    # 会话操作
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_session(self, session_id: str) -> dict | None:
        """获取单个会话信息"""
        ...

    @abstractmethod
    async def get_sessions_for_user(self, user_id: int) -> list[dict]:
        """获取用户的所有会话，按更新时间降序排列"""
        ...

    @abstractmethod
    async def create_session(
        self, session_id: str, user_id: int, title: str = "",
        role: str = "default", model_name: str = "deepseek-chat",
    ) -> None:
        """创建新会话"""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """删除会话（级联删除关联消息）"""
        ...

    @abstractmethod
    async def rename_session(self, session_id: str, title: str) -> None:
        """重命名会话"""
        ...

    @abstractmethod
    async def update_session_role(self, session_id: str, role: str) -> None:
        """更新会话的角色设定"""
        ...

    @abstractmethod
    async def update_session_model(self, session_id: str, model_name: str) -> None:
        """更新会话使用的模型名称"""
        ...

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_messages_for_role(
        self, session_id: str, role_name: str
    ) -> list[dict]:
        """获取某个会话中特定角色的消息历史（按时间升序）"""
        ...

    @abstractmethod
    async def save_message(
        self, session_id: str, role: str, content: str, role_name: str = "default",
        prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0,
    ) -> None:
        """保存一条消息并更新会话的更新时间，可附带 token 用量"""
        ...

    @abstractmethod
    async def get_all_messages(self, session_id: str) -> list[dict]:
        """获取某个会话的所有消息（不区分角色，按时间升序）"""
        ...

    @abstractmethod
    async def auto_title(self, session_id: str, message: str) -> None:
        """若会话尚无标题，则取用户消息前20字符作为自动标题"""
        ...

    # ------------------------------------------------------------------
    # Token 统计
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_session_tokens(self, session_id: str) -> dict:
        """获取会话累计 token 用量"""
        ...

    @abstractmethod
    async def update_session_tokens(self, session_id: str) -> None:
        """重新计算并更新会话的 token 累计值"""
        ...

    # ------------------------------------------------------------------
    # 搜索
    # ------------------------------------------------------------------

    @abstractmethod
    async def search_messages(self, user_id: int, keyword: str) -> list[dict]:
        """搜索用户所有会话中的消息，按关键词匹配"""
        ...

    # ------------------------------------------------------------------
    # 预设操作
    # ------------------------------------------------------------------

    @abstractmethod
    async def create_preset(
        self, user_id: int, name: str, description: str, system_prompt: str
    ) -> dict:
        """创建用户自定义预设"""
        ...

    @abstractmethod
    async def get_presets_for_user(self, user_id: int) -> list[dict]:
        """获取用户的所有自定义预设"""
        ...

    @abstractmethod
    async def get_preset_by_name(self, name: str, user_id: int) -> dict | None:
        """根据名称查找用户预设"""
        ...

    @abstractmethod
    async def update_preset(
        self, preset_id: int, user_id: int,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
    ) -> dict | None:
        """更新用户预设"""
        ...

    @abstractmethod
    async def delete_preset(self, preset_id: int, user_id: int) -> bool:
        """删除用户预设"""
        ...

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """检查数据库是否可用"""
        ...
