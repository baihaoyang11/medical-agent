"""
答案生成 Skill — 封装 DeepSeek LLM 调用 + 医学提示词工程

复用原 agent.py 第115-126行的 prompt 模板和 LLM 调用逻辑
Agent 在收集完检索结果后调用此 Skill 生成最终答案
"""

from typing import Any, Dict, Optional
from langchain_openai import ChatOpenAI

from core.base_skill import BaseSkill, SkillResult


# 医学助手系统提示词（从原 agent.py 提取并增强）
SYSTEM_PROMPT = (
    "你是一个非常得力的医学助手，可以基于从数据库中检索出的信息来回答问题。"
    "请遵循以下原则：\n"
    "1. 优先使用检索到的数据库内容回答\n"
    "2. 如果检索信息不足，可以结合你的医学知识补充\n"
    "3. 不知道的时候坦诚承认不了解，不要编造不真实的信息\n"
    "4. 涉及诊断建议时，提醒用户咨询专业医生\n"
    "5. 回答要专业、准确、易懂"
)

USER_PROMPT_TEMPLATE = (
    "User:利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题，"
    "具体的问题介于<question>和</question>之间。"
    "如果提供的信息为空，则按照你的经验知识来给出尽可能严谨准确的回答，"
    "不知道的时候坦诚的承认不了解，不要编造不真实的信息。\n"
    "<context>\n{context}\n</context>\n"
    "<question>\n{query}\n</question>"
)


class AnswerGenerationSkill(BaseSkill):
    """基于检索上下文 + DeepSeek LLM 的答案生成"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__()
        self._llm = llm
        self._name = "answer_generation"
        self._description = (
            "基于检索到的医学文献上下文，使用 DeepSeek-Chat 大语言模型生成专业、准确的医学答案。"
            "会自动应用医学提示词工程，确保答案引用数据库内容、防止幻觉、标注信息来源。"
            "在所有检索完成后，最终调用此工具生成用户可读的答案。"
        )
        self._parameters = {
            "query": {
                "type": "string",
                "description": "用户的原始医学问题",
            },
            "context": {
                "type": "string",
                "description": "从检索系统中获取的相关医学文献上下文",
            },
        }

    def execute(
        self,
        query: str = "",
        context: str = "",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成医学答案

        Args:
            query: 用户问题
            context: 检索到的上下文
            system_prompt: 可选的自定义系统提示词

        Returns:
            标准化的 SkillResult，data 中包含生成的答案
        """
        if not query:
            return SkillResult.fail("query 参数不能为空")

        try:
            sys_prompt = system_prompt or SYSTEM_PROMPT
            user_prompt = USER_PROMPT_TEMPLATE.format(
                context=context if context else "（无检索结果，请依据医学知识回答）",
                query=query,
            )

            full_prompt = f"System:{sys_prompt}\n\n{user_prompt}"
            response = self._llm.invoke(full_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            return SkillResult.ok(
                data=answer,
                model="deepseek-chat",
                context_length=len(context),
            )

        except Exception as e:
            return SkillResult.fail(f"答案生成失败: {str(e)}", query=query)
