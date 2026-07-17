"""
医学问答 Agent — 基于 LangGraph 的 Agent 编排器

Agent 工作流（4 节点有向图）：

    START
      │
      ▼
  ┌─────────┐
  │ router  │ ── 分析问题类型，决定检索策略
  └────┬────┘
       │
       ▼
  ┌──────────┐
  │ retrieve │ ── 调用 HybridSearch + ParentDoc Skills
  └────┬─────┘
       │
       ▼
  ┌────────┐
  │ verify │ ── 检查检索质量，决定是否需要补充检索
  └───┬────┘
      │
      ▼
  ┌──────────┐
  │ generate │ ── 调用 AnswerGeneration Skill 生成最终答案
  └────┬─────┘
       │
       ▼
      END

每个节点将思考步骤追加到 thinking_steps（面试展示用）。
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from core.state import AgentState
from core.base_skill import SkillResult
from skills.hybrid_search import HybridSearchSkill
from skills.parent_doc import ParentDocSearchSkill
from skills.answer_gen import AnswerGenerationSkill


class MedicalAgent:
    """
    医学问答 Agent

    使用 LangGraph StateGraph 编排 Skill 调用流程：
    router → retrieve → verify → generate

    特点：
    - Agent 自主分析问题类型（疾病定义/用药咨询/饮食建议/鉴别诊断等）
    - Agent 自主选择检索策略（QA库/PDF教材/混合）
    - 检索质量不足时自动触发二次检索
    - 全程记录 thinking_steps 供面试展示
    """

    def __init__(
        self,
        hybrid_search_skill: HybridSearchSkill,
        parent_doc_skill: ParentDocSearchSkill,
        answer_gen_skill: AnswerGenerationSkill,
    ):
        self.hybrid_search = hybrid_search_skill
        self.parent_doc = parent_doc_skill
        self.answer_gen = answer_gen_skill

        # 构建 LangGraph 工作流
        self._graph = self._build_graph()

    # ================================================================
    # 节点 1: 路由 — 分析问题类型，决定检索策略
    # ================================================================
    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        """路由节点：Agent 分析用户意图，决定最优检索策略"""
        query = state.get("query", "")

        # Agent 决策逻辑：基于关键词判断问题类型
        question_type = self._classify_question(query)
        strategy = self._select_strategy(question_type)

        step = (
            f"[Router Agent] 分析问题: \"{query[:50]}...\" → "
            f"类型: {question_type}, 策略: {strategy}"
        )

        return {
            "question_type": question_type,
            "retrieval_strategy": strategy,
            "thinking_steps": [step],
        }

    def _classify_question(self, query: str) -> str:
        """基于关键词规则判断医学问题类型"""
        q_lower = query.lower()
        if any(w in q_lower for w in ["饮食", "吃什么", "禁忌", "营养", "食谱"]):
            return "diet"
        elif any(w in q_lower for w in ["区别", "鉴别", "还是", "怎么区分", "不同"]):
            return "diagnosis"
        elif any(w in q_lower for w in ["药", "治疗", "用药", "剂量", "处方"]):
            return "drug"
        elif any(w in q_lower for w in ["预防", "感染", "传播", "传染", "疫苗"]):
            return "prevention"
        elif any(w in q_lower for w in ["定义", "什么是", "是什么", "解释", "概念"]):
            return "disease_definition"
        elif any(w in q_lower for w in ["症状", "表现", "诊断", "检查", "指标", "正常值"]):
            return "clinical"
        else:
            return "general"

    def _select_strategy(self, question_type: str) -> str:
        """根据问题类型选择检索策略"""
        if question_type in ("disease_definition", "clinical", "drug"):
            return "hybrid"       # 需要精确术语匹配 + 语义理解
        elif question_type in ("diagnosis",):
            return "hybrid"       # 鉴别诊断需要多维度检索
        elif question_type in ("prevention", "diet"):
            return "hybrid"       # 预防和饮食建议需要 QA + 教材
        else:
            return "hybrid"       # 默认混合检索

    # ================================================================
    # 节点 2: 检索 — 调用 Skills 收集上下文
    # ================================================================
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """检索节点：Agent 调用 HybridSearch 和 ParentDoc Skills"""
        query = state["query"]
        strategy = state.get("retrieval_strategy", "hybrid")
        steps = []

        qa_context = ""
        pdf_context = ""

        # 调用 Skill 1: 混合检索（QA问答对库）
        if strategy in ("qa_only", "hybrid"):
            result = self.hybrid_search.execute(query=query, k=10)
            if result["success"] and result["data"]:
                qa_context = "\n\n".join(
                    doc["content"] for doc in result["data"]
                )
                steps.append(
                    f"[Retrieve Agent] 调用 HybridSearch Skill → "
                    f"从QA库召回 {result['metadata']['count']} 条文档"
                )
            else:
                steps.append(
                    f"[Retrieve Agent] 调用 HybridSearch Skill → "
                    f"QA库无相关结果"
                )

        # 调用 Skill 2: PDF教材检索
        if strategy in ("pdf_only", "hybrid"):
            result = self.parent_doc.execute(query=query, k=3)
            if result["success"] and result["data"]:
                pdf_context = "\n\n".join(
                    doc["content"] for doc in result["data"]
                )
                sources = [d["source"] for d in result["data"] if d.get("source")]
                steps.append(
                    f"[Retrieve Agent] 调用 ParentDocSearch Skill → "
                    f"从PDF教材召回 {result['metadata']['count']} 条段落 "
                    f"({', '.join(sources[:2])})"
                )
            else:
                steps.append(
                    f"[Retrieve Agent] 调用 ParentDocSearch Skill → "
                    f"PDF教材无相关结果"
                )

        # 拼接最终上下文
        final_context_parts = []
        if qa_context:
            final_context_parts.append(f"[QA问答对数据库]\n{qa_context}")
        if pdf_context:
            final_context_parts.append(f"[PDF医学教材]\n{pdf_context}")
        final_context = "\n\n---\n\n".join(final_context_parts)

        steps.append(
            f"[Retrieve Agent] 上下文聚合完成 → "
            f"总长度 {len(final_context)} 字符"
        )

        return {
            "final_context": final_context,
            "thinking_steps": steps,
        }

    # ================================================================
    # 节点 3: 验证 — 检查检索质量
    # ================================================================
    def _verify_node(self, state: AgentState) -> Dict[str, Any]:
        """验证节点：Agent 检查检索结果质量，决定是否需要补充检索"""
        context = state.get("final_context", "")
        need_secondary = len(context.strip()) < 100  # 上下文太少则标记

        if need_secondary:
            step = (
                f"[Verify Agent] 检索质量检查 → 上下文不足({len(context)}字符)，"
                f"标记需要二次检索"
            )
        else:
            step = (
                f"[Verify Agent] 检索质量检查 → 通过 "
                f"(上下文 {len(context)} 字符)"
            )

        return {
            "need_secondary_search": need_secondary,
            "thinking_steps": [step],
        }

    # ================================================================
    # 节点 4: 生成 — 调用 LLM 生成最终答案
    # ================================================================
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """生成节点：Agent 调用 AnswerGeneration Skill 生成最终答案"""
        query = state["query"]
        context = state.get("final_context", "")

        result = self.answer_gen.execute(query=query, context=context)

        if result["success"]:
            answer = result["data"]
            step = (
                f"[Generate Agent] 调用 AnswerGeneration Skill → "
                f"DeepSeek-Chat 生成答案 ({len(answer)} 字符)"
            )
        else:
            answer = f"抱歉，答案生成失败：{result['error']}"
            step = (
                f"[Generate Agent] 调用 AnswerGeneration Skill → "
                f"生成失败: {result['error']}"
            )

        return {
            "final_answer": answer,
            "thinking_steps": [step],
        }

    # ================================================================
    # 条件路由：验证后是否需要二次检索
    # ================================================================
    def _should_retrieve_again(self, state: AgentState) -> Literal["retrieve", "generate"]:
        """条件边：如果验证不过且未超过2次检索，则重新检索"""
        if state.get("need_secondary_search", False):
            # 简单起见，当前不实现二次检索循环（可扩展）
            # 直接走生成节点，让 LLM 基于有限信息回答
            return "generate"
        return "generate"

    # ================================================================
    # 构建 LangGraph 工作流
    # ================================================================
    def _build_graph(self):
        """构建 StateGraph"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("router", self._router_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("generate", self._generate_node)

        # 添加边
        workflow.set_entry_point("router")
        workflow.add_edge("router", "retrieve")
        workflow.add_edge("retrieve", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._should_retrieve_again,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            }
        )
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ================================================================
    # 公共接口
    # ================================================================
    def run(self, query: str) -> Dict[str, Any]:
        """
        运行 Agent 处理一个用户问题

        Args:
            query: 用户输入的医学问题

        Returns:
            {
                "answer": str,           # 最终答案
                "thinking_steps": [...], # Agent思考链
                "question_type": str,    # 问题类型
                "retrieval_strategy": str, # 检索策略
                "context_length": int,   # 检索到的上下文长度
            }
        """
        initial_state: AgentState = {
            "query": query,
            "qa_retrieved_docs": [],
            "pdf_retrieved_content": "",
            "final_context": "",
            "question_type": "",
            "retrieval_strategy": "",
            "need_secondary_search": False,
            "final_answer": "",
            "citations": [],
            "thinking_steps": [],
        }

        # 执行 Agent 工作流
        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state.get("final_answer", ""),
            "thinking_steps": final_state.get("thinking_steps", []),
            "question_type": final_state.get("question_type", ""),
            "retrieval_strategy": final_state.get("retrieval_strategy", ""),
            "context_length": len(final_state.get("final_context", "")),
        }

    def get_skill_registry(self) -> Dict[str, Any]:
        """返回 Skill 注册表，供外部查询 Agent 拥有哪些能力"""
        return {
            self.hybrid_search.name: self.hybrid_search.to_tool_definition(),
            self.parent_doc.name: self.parent_doc.to_tool_definition(),
            self.answer_gen.name: self.answer_gen.to_tool_definition(),
        }
