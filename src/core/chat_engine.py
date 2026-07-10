"""
对话引擎
负责 LLM 构建、对话链调用、流式/非流式响应生成
整合 SessionManager、PresetManager 和 ConfigManager
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from langchain_classic.chains import ConversationChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.core.config_manager import ConfigManager, get_config
from src.core.preset_manager import PresetManager, get_preset_manager
from src.core.session_manager import SessionManager
from src.storage.base import StorageBackend

logger = logging.getLogger("chatbot")


class ChatEngine:
    """对话引擎，封装 LLM 调用与流式输出逻辑"""

    def __init__(
        self,
        session_manager: SessionManager,
        preset_manager: PresetManager | None = None,
        config: ConfigManager | None = None,
        storage: StorageBackend | None = None,
    ):
        self._sessions = session_manager
        self._presets = preset_manager or get_preset_manager()
        self._config = config or get_config()
        self._storage = storage

    # ------------------------------------------------------------------
    # LLM 构建
    # ------------------------------------------------------------------

    def build_llm(self, model_name: Optional[str] = None) -> ChatOpenAI:
        """根据模型名称构建 ChatOpenAI 实例

        参数:
            model_name: 模型名称，为 None 时使用默认模型

        返回:
            配置好的 ChatOpenAI 实例

        异常:
            ValueError: 如果模型未在注册表中配置或 API Key 未设置
        """
        name = model_name or self._config.default_model
        cfg = self._config.get_model_config(name)
        if not cfg:
            available = [m.get("name", "?") for m in self._config.models]
            raise ValueError(f"未知模型 '{name}'。可用: {available}")

        api_key = cfg.get("api_key", "")
        base_url = cfg.get("base_url", "")
        if not api_key:
            raise ValueError(f"模型 '{name}' 的 API Key 未设置，请检查 .env")

        return ChatOpenAI(
            model=name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=self._config.llm_temperature,
        )

    # ------------------------------------------------------------------
    # 非流式聊天
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: int,
        role: str = "default",
        model: Optional[str] = None,
    ) -> dict:
        """发送消息并返回完整 AI 回复（非流式）

        返回:
            {"response": str, "session_id": str}
        """
        await self._sessions.verify_ownership(session_id, user_id)
        await self._sessions.load_memory(session_id, role)

        model_name = model or self._config.default_model
        await self._sessions.update_model(session_id, model_name)

        llm = self.build_llm(model)
        memory = self._sessions.get_memory(session_id, role)

        chain = ConversationChain(llm=llm, memory=memory, verbose=False)

        await self._sessions.save_message(session_id, "user", message, role)
        result = await chain.ainvoke({"input": message})
        response_text = result.get("response", "")
        await self._sessions.save_message(session_id, "assistant", response_text, role)

        return {"response": response_text, "session_id": session_id}

    # ------------------------------------------------------------------
    # 流式聊天（SSE）
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        user_id: int,
        role: str = "default",
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """发送消息并以 SSE 格式流式返回 AI 回复

        生成:
            SSE 格式的事件字符串（"data: {...}\n\n"）
        """
        try:
            await self._sessions.verify_ownership(session_id, user_id)

            # 获取角色的系统提示词（内置优先，其次用户自定义）
            if self._presets.exists(role):
                system_prompt = self._presets.get_system_prompt(role)
            elif self._storage:
                preset = await self._storage.get_preset_by_name(role, user_id)
                system_prompt = preset["system_prompt"] if preset else self._presets.get_system_prompt("default")
            else:
                system_prompt = self._presets.get_system_prompt("default")

            # 更新会话角色和模型
            model_name = model or self._config.default_model
            await self._sessions.update_model(session_id, model_name)

            session = await self._sessions.get_session(session_id)
            if session and session.get("role") != role:
                await self._sessions.update_role(session_id, role, user_id)

            # 构建 LLM 和消息链
            llm = self.build_llm(model)
            memory = await self._sessions.load_memory(session_id, role)

            messages = [SystemMessage(content=system_prompt)]
            messages.extend(memory.chat_memory.messages)
            messages.append(HumanMessage(content=message))

            # 保存用户消息
            await self._sessions.save_message(session_id, "user", message, role)

            # 流式生成
            full_response: str = ""
            async for chunk in llm.astream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    full_response += content
                    yield (
                        "data: "
                        + json.dumps(
                            {"content": content, "session_id": session_id},
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

            # 保存完整回复和上下文
            self._sessions.save_context(session_id, role, message, full_response)
            await self._sessions.save_message(session_id, "assistant", full_response, role)

            yield (
                "data: "
                + json.dumps({"done": True, "session_id": session_id})
                + "\n\n"
            )

        except asyncio.CancelledError:
            logger.info("SSE 流被客户端取消")
        except Exception:
            logger.exception("流式聊天出错")
            yield f"data: {json.dumps({'error': '服务器内部错误'})}\n\n"

    # ------------------------------------------------------------------
    # 可用模型列表
    # ------------------------------------------------------------------

    def list_models(self) -> dict:
        """返回可用模型列表及默认模型"""
        return {
            "models": [m.get("name", "?") for m in self._config.models],
            "default": self._config.default_model,
        }
