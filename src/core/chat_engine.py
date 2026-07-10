"""
对话引擎
负责 LLM 构建、对话链调用、流式/非流式响应生成
整合 SessionManager、PresetManager 和 ConfigManager
"""

import asyncio
import json
import logging
import time
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
        """根据模型名称构建 ChatOpenAI 实例"""
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
            model_kwargs={"stream_options": {"include_usage": True}},
        )

    # ------------------------------------------------------------------
    # 带超时和重试的 LLM 流式调用
    # ------------------------------------------------------------------

    async def _stream_with_retry(
        self, llm: ChatOpenAI, messages: list, user_id: int, session_id: str, model_name: str,
    ) -> AsyncGenerator[dict, None]:
        """流式调用 LLM，支持超时和指数退避重试。

        Yields:
            dict: {"type": "token", "content": str} 或 {"type": "usage", "data": dict}
        """
        timeout_sec = self._config.llm_timeout
        max_retries = self._config.llm_max_retries
        backoff = self._config.llm_retry_backoff

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                t_start = time.monotonic()
                usage: dict[str, int] = {}

                async with asyncio.timeout(timeout_sec):
                    async for chunk in llm.astream(messages):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield {"type": "token", "content": content}
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            um = chunk.usage_metadata
                            usage = {
                                "prompt_tokens": um.get("input_tokens", 0),
                                "completion_tokens": um.get("output_tokens", 0),
                                "total_tokens": um.get("total_tokens", 0),
                            }

                duration = round(time.monotonic() - t_start, 2)
                logger.info(
                    "LLM 调用成功",
                    extra={"user_id": user_id, "session_id": session_id,
                           "model": model_name, "duration": duration, "attempt": attempt + 1},
                )
                if usage:
                    yield {"type": "usage", "data": usage}
                return

            except asyncio.TimeoutError:
                last_error = f"超时（{timeout_sec}s）"
                if attempt < max_retries:
                    wait = backoff ** attempt
                    logger.warning(
                        f"LLM 调用超时，{wait}s 后重试",
                        extra={"user_id": user_id, "session_id": session_id,
                               "model": model_name, "attempt": attempt + 1, "retry_wait": wait},
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"LLM 调用超时，已达最大重试次数",
                        extra={"user_id": user_id, "session_id": session_id,
                               "model": model_name, "max_retries": max_retries},
                    )

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    wait = backoff ** attempt
                    logger.warning(
                        f"LLM 调用失败，{wait}s 后重试: {e}",
                        extra={"user_id": user_id, "session_id": session_id,
                               "model": model_name, "attempt": attempt + 1, "retry_wait": wait},
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"LLM 调用失败，已达最大重试次数: {e}",
                        extra={"user_id": user_id, "session_id": session_id, "model": model_name},
                    )

        raise RuntimeError(f"LLM 调用失败: {last_error}")

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
        """发送消息并返回完整 AI 回复（非流式）"""
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
        """发送消息并以 SSE 格式流式返回 AI 回复（支持超时和重试）"""
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

            # 流式生成（带超时和重试）
            full_response: str = ""
            usage: dict[str, int] = {}
            async for event in self._stream_with_retry(llm, messages, user_id, session_id, model_name):
                if event["type"] == "token":
                    full_response += event["content"]
                    yield (
                        "data: "
                        + json.dumps(
                            {"content": event["content"], "session_id": session_id},
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                elif event["type"] == "usage":
                    usage = event["data"]

            # 保存完整回复（含 token 用量）并更新会话统计
            self._sessions.save_context(session_id, role, message, full_response)
            await self._sessions.save_message(
                session_id, "assistant", full_response, role,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            if usage:
                await self._sessions.update_session_tokens(session_id)

            done_payload = {"done": True, "session_id": session_id}
            if usage:
                done_payload["usage"] = usage
            yield "data: " + json.dumps(done_payload, ensure_ascii=False) + "\n\n"

        except asyncio.CancelledError:
            logger.info("SSE 流被客户端取消")
        except Exception:
            logger.exception("流式聊天出错")
            yield f"data: {json.dumps({'error': '服务器内部错误'})}\n\n"

    # ------------------------------------------------------------------
    # 多模型并行对比
    # ------------------------------------------------------------------

    async def parallel_chat_stream(
        self,
        message: str,
        model_names: list[str],
        session_id: str,
        user_id: int,
        role: str = "default",
    ) -> AsyncGenerator[str, None]:
        """并行调用多个模型，流式输出对比结果。

        每个模型的 token 以 SSE 格式产出，包含 model 标签。
        所有模型结束后发送 all_done 事件。
        """
        try:
            await self._sessions.verify_ownership(session_id, user_id)

            # 解析 system prompt
            if self._presets.exists(role):
                system_prompt = self._presets.get_system_prompt(role)
            elif self._storage:
                preset = await self._storage.get_preset_by_name(role, user_id)
                system_prompt = preset["system_prompt"] if preset else self._presets.get_system_prompt("default")
            else:
                system_prompt = self._presets.get_system_prompt("default")

            # 加载对话历史
            memory = await self._sessions.load_memory(session_id, role)
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(memory.chat_memory.messages)
            messages.append(HumanMessage(content=message))

            # 保存用户消息（只保存一次）
            await self._sessions.save_message(session_id, "user", message, role)

            # 使用 Queue 合并多个并行流
            queue: asyncio.Queue = asyncio.Queue()
            total = len(model_names)

            async def _stream_one_model(model_name: str) -> None:
                """单个模型的流式调用"""
                try:
                    llm = self.build_llm(model_name)
                    t_start = time.monotonic()
                    full_response = ""
                    async for event in self._stream_with_retry(
                        llm, messages, user_id, session_id, model_name,
                    ):
                        if event["type"] == "token":
                            full_response += event["content"]
                            await queue.put({
                                "model": model_name,
                                "content": event["content"],
                            })
                    duration = round(time.monotonic() - t_start, 2)
                    await queue.put({
                        "model": model_name,
                        "done": True,
                        "response": full_response,
                        "duration": duration,
                    })
                    # 保存助手消息
                    await self._sessions.save_message(
                        session_id, "assistant", full_response, model_name,
                    )
                    logger.info(
                        "并行LLM调用成功",
                        extra={"user_id": user_id, "session_id": session_id,
                               "model": model_name, "duration": duration},
                    )
                except Exception as e:
                    logger.error(
                        f"并行LLM调用失败: {e}",
                        extra={"user_id": user_id, "session_id": session_id, "model": model_name},
                    )
                    await queue.put({
                        "model": model_name,
                        "error": str(e),
                        "done": True,
                    })

            # 启动所有并行任务
            tasks = [asyncio.create_task(_stream_one_model(m)) for m in model_names]

            # 从队列读取事件，直到所有模型完成
            done_count = 0
            while done_count < total:
                event = await queue.get()
                if event.get("done"):
                    done_count += 1
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

            yield "data: " + json.dumps({"all_done": True}) + "\n\n"

        except asyncio.CancelledError:
            logger.info("并行SSE流被客户端取消")
        except Exception:
            logger.exception("并行聊天出错")
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
