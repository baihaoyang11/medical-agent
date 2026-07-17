"""
Skill 基类 — 所有可插拔能力的统一接口

每个 Skill 都是一个 Agent 可以调用的"工具"，有明确的：
- name: 技能名称（Agent 用来识别）
- description: 技能描述（Agent 用来决策何时调用）
- parameters: 输入参数 JSON Schema
- execute(): 执行逻辑，返回标准化 dict
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSkill(ABC):
    """Skill 基类，所有 Skill 必须继承此类"""

    def __init__(self):
        self._name: str = self.__class__.__name__
        self._description: str = ""
        self._parameters: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        """返回 OpenAI Function Calling 兼容的 JSON Schema"""
        return {
            "type": "object",
            "properties": self._parameters,
            "required": list(self._parameters.keys()),
        }

    def to_tool_definition(self) -> Dict[str, Any]:
        """导出为标准的 Tool Definition（兼容 LangChain / OpenAI）"""
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": self.parameters,
            }
        }

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行 Skill，返回标准化结果

        Returns:
            {
                "success": bool,
                "data": Any,
                "error": str | None,
                "metadata": dict   # 额外的元信息（如耗时、来源等）
            }
        """
        raise NotImplementedError


class SkillResult:
    """Skill 执行结果的标准化封装"""

    @staticmethod
    def ok(data: Any = None, **metadata) -> Dict[str, Any]:
        return {
            "success": True,
            "data": data,
            "error": None,
            "metadata": metadata,
        }

    @staticmethod
    def fail(error: str, **metadata) -> Dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "error": error,
            "metadata": metadata,
        }
