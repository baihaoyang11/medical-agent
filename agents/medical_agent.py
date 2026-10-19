"""医学问答 Agent — 基于 LangGraph 的编排器（企业化改造版）

工作流（4 节点 + 1 安全通道）：

    START
      │
      ▼
   router ── 急症? ──► safety（直接给就医指引，不检索不生成）
      │
      ▼
   retrieve ── 按策略调 HybridSearch / ParentDoc，失败降级
      │
      ▼
   verify ── 质量判定（条数 / 长度 / 关键词覆盖）
      │
      ├── 不达标且未超次数 ──► retrieve（改写 query 重试）
      ▼
   generate ── 带引用编号生成答案
      │
      ▼
     END

相对旧版的四类改动：
1. 路由真的分流：策略表决定 qa_k / pdf_k / 上下文预算，不再是六个分支返回同一个值
2. 条件边真的回退：retry_count + query 改写 + 结果去重，最多 MAX_RETRIES 次
3. 质量判定从"字符数 > 100"升级为三维信号（见 core/quality.py）
4. 补上 citations 与 checkpointer：引用可追溯，状态可通过 thread_id 恢复
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from core.citations import build_citations, render_for_prompt
from core.context import compose_context, dedupe
from core.observability import log, span
from core.quality import assess
from core.rerank import Reranker, build_reranker
from core.retry import call_with_retry
from core.rewrite import rewrite
from core.routing import resolve
from core.state import AgentState

# 二次检索最多再补 2 次（总检索 3 次）
MAX_RETRIES = 2

EMERGENCY_REPLY = (
    "你描述的情况可能属于需要紧急处理的情况。请立即拨打 120 或尽快前往最近的急诊科，"
    "不要等待线上回复，也不要自行用药。如果身边有人，请让对方陪同。"
    "本系统只能提供一般性医学知识，不能替代医生的现场判断。"
)

DISCLAIMER = "以上内容仅供参考，不能替代专业医生的诊断和治疗建议；如有不适请及时就医。"


class MedicalAgent:
    """医学问答 Agent，使用 LangGraph StateGraph 编排 Skill 调用。"""

    def __init__(
        self,
        hybrid_search_skill: Any,
        parent_doc_skill: Any,
        answer_gen_skill: Any,
        *,
        max_retries: int = MAX_RETRIES,
        checkpointer: Any | None = None,
        reranker: Reranker | None = None,
        retriever_routing: bool | None = None,
        retriever_default: str | None = None,
    ) -> None:
        self.hybrid_search = hybrid_search_skill
        self.parent_doc = parent_doc_skill
        self.answer_gen = answer_gen_skill
        self.max_retries = max_retries
        self._checkpointer = checkpointer or MemorySaver()

        from config import config as _cfg

        # 检索器路由默认跟随配置（config.RETRIEVER_ROUTING，默认关闭）。
        # 显式传参优先，测试里可以直接开关而不改环境变量。
        if retriever_routing is None:
            retriever_routing = bool(_cfg.RETRIEVER_ROUTING)
        self._retriever_routing = retriever_routing
        # 全局默认检索器：路由关闭时所有请求用它，路由打开时只用作兜底
        self._retriever_default = retriever_default or getattr(_cfg, "RETRIEVER_DEFAULT", "hybrid")
        # 不传就用默认策略（config 里可改），传对象则完全接管
        if reranker is None:
            reranker = build_reranker(
                _cfg.RERANK_STRATEGY,
                model_path=_cfg.RERANK_MODEL_PATH,
                weight=_cfg.RERANK_WEIGHT,
            )
        self._reranker = reranker
        self._graph = self._build_graph()

    # ================================================================
    # 节点 1：路由
    # ================================================================
    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        decision = resolve(
            query,
            retriever_routing=self._retriever_routing,
            default_retriever=self._retriever_default,
        )

        if decision.emergency:
            step = f"[Router] 命中急症关键词 {list(decision.matched)} → 走安全通道"
        else:
            step = (
                f"[Router] 类型={decision.question_type} 策略={decision.strategy} "
                f"(qa_k={decision.params.qa_k}, pdf_k={decision.params.pdf_k}, "
                f"检索器={decision.retriever})"
            )

        log("route", **decision.to_trace())

        return {
            "question_type": decision.question_type,
            "retrieval_strategy": decision.strategy,
            "retriever_mode": decision.retriever,
            "emergency": decision.emergency,
            "route_matched": list(decision.matched),
            "effective_query": rewrite(query, 1),
            "used_queries": [query],
            "retry_count": 0,
            "thinking_steps": [step],
        }

    # ================================================================
    # 节点：安全通道
    # ================================================================
    def _safety_node(self, state: AgentState) -> Dict[str, Any]:
        return {
            "final_answer": f"{EMERGENCY_REPLY}\n\n{DISCLAIMER}",
            "citations": [],
            "degraded": False,
            "thinking_steps": ["[Safety] 急症场景，跳过检索与生成，直接输出就医指引"],
        }

    # ================================================================
    # 节点 2：检索
    # ================================================================
    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("effective_query") or state["query"]
        retry_count = state.get("retry_count", 0)
        decision = resolve(
            state["query"],
            retriever_routing=self._retriever_routing,
            default_retriever=self._retriever_default,
        )
        params = decision.params
        # 以 router 节点写下的一致决策为准（重试时不重新分类，避免同一请求内两种模式）
        mode = state.get("retriever_mode") or decision.retriever
        steps: list[str] = []

        qa_docs: list[dict[str, Any]] = []
        pdf_docs: list[dict[str, Any]] = []

        if params.use_qa:
            with span("skill.hybrid_search", k=params.qa_k, mode=mode, attempt=retry_count + 1):
                result = call_with_retry(
                    self.hybrid_search.execute,
                    query=query,
                    k=params.qa_k,
                    mode=mode,
                    op_name="hybrid_search",
                )
            if result.get("success"):
                qa_docs = list(result.get("data") or [])
                method = (result.get("metadata") or {}).get("method")
                steps.append(f"[Retrieve] HybridSearch 命中 {len(qa_docs)} 条（{method}，mode={mode}）")
            else:
                # 检索失败降级为"没有 QA 上下文"，不让整条链路 500
                steps.append(f"[Retrieve] HybridSearch 失败已降级：{result.get('error')}")

        if params.use_pdf:
            with span("skill.parent_doc_search", k=params.pdf_k, mode=mode, attempt=retry_count + 1):
                result = call_with_retry(
                    self.parent_doc.execute,
                    query=query,
                    k=params.pdf_k,
                    mode=mode,
                    op_name="parent_doc_search",
                )
            if result.get("success"):
                pdf_docs = list(result.get("data") or [])
                sources = sorted({d.get("source") for d in pdf_docs if d.get("source")})
                steps.append(
                    f"[Retrieve] ParentDoc 命中 {len(pdf_docs)} 段父文档（{', '.join(sources) or '无来源'}）"
                )
            else:
                steps.append(f"[Retrieve] ParentDoc 失败已降级：{result.get('error')}")

        merged = dedupe(qa_docs + pdf_docs)

        # 融合只用了排名信息，尺度不可比；重排把"真正答这个问题"的段落提到前面，
        # 上下文预算按重要性分配才有意义（超预算时丢的是尾部）。
        if merged and getattr(self._reranker, "name", "noop") != "noop":
            with span("rerank", strategy=self._reranker.name, docs=len(merged)):
                merged = self._reranker.rerank(query, merged)
            mark = "（已降级）" if getattr(self._reranker, "degraded", False) else ""
            steps.append(f"[Retrieve] {self._reranker.name} 重排 {len(merged)} 条{mark}")

        context = compose_context(merged, params.max_context_chars)

        steps.append(
            f"[Retrieve] 去重后 {len(merged)} 条，上下文 {len(context)} 字符"
            f"（预算 {params.max_context_chars}）"
        )

        return {
            "qa_docs": qa_docs,
            "pdf_docs": pdf_docs,
            "merged_docs": merged,
            "final_context": context,
            "used_queries": [query],
            "thinking_steps": steps,
        }

    # ================================================================
    # 节点 3：质量校验
    # ================================================================
    def _verify_node(self, state: AgentState) -> Dict[str, Any]:
        report = assess(
            state.get("query", ""),
            state.get("merged_docs", []),
            state.get("final_context", ""),
        )
        retry_count = state.get("retry_count", 0)
        need_retry = (not report.passed) and retry_count < self.max_retries

        if report.passed:
            step = (
                f"[Verify] 通过：{report.doc_count} 条 / {report.context_chars} 字符"
                f" / 覆盖 {report.coverage:.0%}"
            )
        elif need_retry:
            step = (
                f"[Verify] 不通过（{'；'.join(report.reasons)}）"
                f"→ 第 {retry_count + 1} 次改写重试"
            )
        else:
            step = f"[Verify] 不通过（{'；'.join(report.reasons)}）且已达重试上限 → 降级生成"

        log(
            "verify",
            passed=report.passed,
            need_retry=need_retry,
            retry_count=retry_count,
            reasons=report.reasons,
        )

        return {
            "quality": report.to_trace(),
            "need_secondary_search": need_retry,
            "thinking_steps": [step],
        }

    # ================================================================
    # 节点 4：生成
    # ================================================================
    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        citations = build_citations(state.get("merged_docs", []))
        context = state.get("final_context", "")

        citation_block = render_for_prompt(citations)
        if citation_block:
            context = f"{context}\n\n[可引用来源]\n{citation_block}"

        with span("skill.answer_generation", citations=len(citations)):
            result = call_with_retry(
                self.answer_gen.execute,
                query=state["query"],
                context=context,
                op_name="answer_generation",
            )

        if result.get("success"):
            answer = str(result.get("data") or "")
            step = f"[Generate] 生成 {len(answer)} 字符，引用 {len(citations)} 处"
        else:
            answer = "抱歉，答案生成暂时不可用，请稍后重试。"
            step = f"[Generate] 生成失败：{result.get('error')}"

        degraded = bool(state.get("need_secondary_search")) or not citations

        return {
            "final_answer": f"{answer}\n\n{DISCLAIMER}",
            "citations": citations,
            "degraded": degraded,
            "thinking_steps": [step],
        }

    # ================================================================
    # 条件边
    # ================================================================
    def _after_route(self, state: AgentState) -> Literal["safety", "retrieve"]:
        return "safety" if state.get("emergency") else "retrieve"

    def _after_verify(self, state: AgentState) -> Literal["retrieve", "generate"]:
        return "retrieve" if state.get("need_secondary_search") else "generate"

    def _bump_retry(self, state: AgentState) -> Dict[str, Any]:
        """进入下一轮检索前：retry_count +1，并改写 query。

        注意：LangGraph 的节点函数不能直接放在条件边返回的路径上，
        这里通过 retrieve 节点的入口统一调用（见 _build_graph 的 retry 节点）。
        """
        next_count = state.get("retry_count", 0) + 1
        new_query = rewrite(state["query"], attempt=next_count + 1)
        log(
            "retry_rewrite",
            attempt=next_count,
            from_query=state.get("effective_query"),
            to_query=new_query,
        )
        return {
            "retry_count": next_count,
            "effective_query": new_query,
            "thinking_steps": [f"[Retrieve] 第 {next_count} 次改写：{new_query}"],
        }

    # ================================================================
    # 构图
    # ================================================================
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("safety", self._safety_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("verify", self._verify_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("retry", self._bump_retry)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router", self._after_route, {"safety": "safety", "retrieve": "retrieve"}
        )
        workflow.add_edge("retrieve", "verify")
        # 不达标 → 先经过 retry 节点计数并改写 query，再回 retrieve
        workflow.add_conditional_edges(
            "verify", self._after_verify, {"retrieve": "retry", "generate": "generate"}
        )
        workflow.add_edge("retry", "retrieve")
        workflow.add_edge("generate", END)
        workflow.add_edge("safety", END)

        return workflow.compile(checkpointer=self._checkpointer)

    # ================================================================
    # 公共接口
    # ================================================================
    def run(self, query: str, thread_id: str | None = None) -> Dict[str, Any]:
        """执行一次问答。

        thread_id 相同则复用检查点（多轮场景）；不传则每次新建，
        避免上一轮状态污染下一轮。
        """
        initial: AgentState = {
            "query": query,
            "thinking_steps": [],
            "citations": [],
            "retry_count": 0,
            "need_secondary_search": False,
            "degraded": False,
        }

        config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}

        with span("agent.run", thread_id=config["configurable"]["thread_id"]):
            final = self._graph.invoke(initial, config)

        return {
            "answer": final.get("final_answer", ""),
            "thinking_steps": final.get("thinking_steps", []),
            "question_type": final.get("question_type", "general"),
            "retrieval_strategy": final.get("retrieval_strategy", "hybrid"),
            "context_length": len(final.get("final_context", "")),
            "citations": final.get("citations", []),
            "quality": final.get("quality", {}),
            "retry_count": final.get("retry_count", 0),
            "emergency": bool(final.get("emergency")),
            "degraded": bool(final.get("degraded")),
        }

    def get_skill_registry(self) -> Dict[str, Any]:
        return {
            self.hybrid_search.name: self.hybrid_search.to_tool_definition(),
            self.parent_doc.name: self.parent_doc.to_tool_definition(),
            self.answer_gen.name: self.answer_gen.to_tool_definition(),
        }
