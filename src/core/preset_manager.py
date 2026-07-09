"""
预设管理器
负责加载和管理角色/人设预设，支持从 YAML 配置文件定义多角色系统提示词
"""

from pathlib import Path

import yaml

from src.models.schemas import PresetInfo


class PresetManager:
    """角色预设管理器，提供预设的加载、查询和列表功能"""

    def __init__(self, presets_path: str = "config/presets.yaml"):
        self._presets_path = Path(presets_path)
        self._presets: dict[str, PresetInfo] = {}
        self._load_presets()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load_presets(self) -> None:
        """从 YAML 文件加载预设并建立索引"""
        if not self._presets_path.exists():
            self._load_defaults()
            return

        with open(self._presets_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for item in data.get("presets", []):
            preset = PresetInfo(
                name=item["name"],
                label=item.get("label", item["name"]),
                emoji=item.get("emoji", "🤖"),
                system_prompt=item.get("system_prompt", ""),
            )
            self._presets[preset.name] = preset

        if not self._presets:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """使用硬编码的默认预设（YAML 文件不存在时的回退方案）"""
        defaults = [
            PresetInfo(
                name="default", label="默认", emoji="🤖",
                system_prompt="你是一个友好的AI助手，请用中文回答用户的问题。",
            ),
            PresetInfo(
                name="teacher", label="老师", emoji="👨‍🏫",
                system_prompt="你是一位耐心的老师，善于用通俗易懂的方式讲解复杂概念。回答要详细、有条理，多用例子说明。",
            ),
            PresetInfo(
                name="programmer", label="程序员", emoji="👨‍💻",
                system_prompt="你是一位资深程序员，回答要简洁、技术导向。涉及代码时要给出完整的示例，注重代码质量。",
            ),
            PresetInfo(
                name="philosopher", label="哲学家", emoji="🧠",
                system_prompt="你是一位哲学家，善于深度思考。回答要富有哲理，引导用户思考问题的本质。",
            ),
            PresetInfo(
                name="friend", label="朋友", emoji="🤝",
                system_prompt="你是一位贴心的朋友，回答要轻松、幽默、随和，像朋友聊天一样自然。",
            ),
        ]
        for p in defaults:
            self._presets[p.name] = p

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_preset(self, name: str) -> PresetInfo:
        """获取指定名称的预设，不存在时返回默认预设"""
        return self._presets.get(name, self._presets["default"])

    def get_system_prompt(self, name: str) -> str:
        """获取指定角色的系统提示词"""
        preset = self.get_preset(name)
        return preset.system_prompt

    def list_presets(self) -> list[PresetInfo]:
        """返回所有可用的预设列表"""
        return list(self._presets.values())

    def list_preset_names(self) -> list[str]:
        """返回所有预设名称列表"""
        return list(self._presets.keys())

    def exists(self, name: str) -> bool:
        """检查指定名称的预设是否存在"""
        return name in self._presets

    def to_choices(self) -> list[tuple[str, str]]:
        """转换为 Gradio Dropdown 所需的 (label, value) 格式"""
        return [
            (f"{p.emoji} {p.label}", p.name)
            for p in self._presets.values()
        ]


# 全局预设管理器单例
_preset_instance: PresetManager | None = None


def get_preset_manager(presets_path: str = "config/presets.yaml") -> PresetManager:
    """获取全局预设管理器单例"""
    global _preset_instance
    if _preset_instance is None:
        _preset_instance = PresetManager(presets_path)
    return _preset_instance
