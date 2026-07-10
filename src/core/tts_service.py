"""
Edge TTS 语音合成服务
使用 Microsoft Edge 免费 TTS 引擎，无需 API Key
"""

import io
import logging

logger = logging.getLogger("chatbot")

# 可选的中文语音
VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",   # 女声（活泼）
    "xiaoyi": "zh-CN-XiaoyiNeural",       # 女声（温柔）
    "yunjian": "zh-CN-YunjianNeural",     # 男声
    "yunxi": "zh-CN-YunxiNeural",         # 男声（新闻）
}


class TTSService:
    """Edge TTS 语音合成服务"""

    async def text_to_speech(
        self, text: str, voice: str = "zh-CN-XiaoxiaoNeural",
    ) -> bytes:
        """将文本转为 MP3 音频，返回二进制数据"""
        import edge_tts

        if not text.strip():
            raise ValueError("文本不能为空")

        # 限制文本长度（TTS 单次请求限制）
        text = text[:2000]

        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)
        except Exception:
            logger.exception("TTS 合成失败")
            raise


# 全局 TTS 服务单例
_tts_instance: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSService()
    return _tts_instance
