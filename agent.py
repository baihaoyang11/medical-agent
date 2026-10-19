"""
医学RAG问答系统 — FastAPI 服务入口（Agent+Skill 架构版）

安全特性：
  - CORS 域名白名单
  - API Key 认证（可选）
  - 单IP请求频率限制
  - 输入长度校验
  - 请求超时治理

架构：
  FastAPI 层 → MedicalAgent (LangGraph) → Skills (HybridSearch / ParentDoc / AnswerGen)

启动方式：
  python agent.py

API 端点：
  POST /          — 基础问答（向后兼容前端）
  POST /api/chat  — 带思考链与引用的完整问答
  GET  /api/skills — 查看所有注册的 Skills
  GET  /api/health — 健康检查
"""

import os
import sys
import time
import hashlib
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# --- 兼容性补丁：解决 Pydantic 2.x 路径问题 ---
try:
    from pydantic import v1 as pydantic_v1
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
    print("✅ Pydantic 兼容性补丁已加载 (V2 -> V1 映射成功)")
except Exception as e:
    print(f"⚠️ 补丁加载失败: {e}")

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import datetime
import shutil

from zhipuai import ZhipuAI
from pymilvus import MilvusClient

from config import config
from model1 import ZhipuAIEmbeddings, create_deepseek_client
from skills.hybrid_search import HybridSearchSkill
from skills.parent_doc import ParentDocSearchSkill
from skills.answer_gen import AnswerGenerationSkill
from agents.medical_agent import MedicalAgent
from core.errors import AppError
from core.observability import bind_trace, current_trace_id, log, new_trace_id

# ================================================================
# 安全工具
# ================================================================

security = HTTPBearer(auto_error=False)

# 内存限流器：{ip: [(timestamp, requests)]}
rate_limit_store: dict = defaultdict(list)

# 进程内指标。单 worker 下准确；多 worker 需要接 Prometheus 聚合，
# 这里的定位是"本地可观测"，不是线上监控系统。
METRICS: dict = {
    "requests_total": 0,
    "errors_total": 0,
    "timeouts_total": 0,
    "rate_limited_total": 0,
    "emergency_total": 0,
    "degraded_total": 0,
}


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_rate_limit(ip: str) -> bool:
    """检查单 IP 是否超过限流阈值（滑动窗口）。"""
    if not config.RATE_LIMIT_ENABLED:
        return True
    now = time.time()
    window_start = now - config.RATE_LIMIT_WINDOW

    # 清理过期记录
    requests = rate_limit_store[ip]
    rate_limit_store[ip] = [t for t in requests if t > window_start]

    if len(rate_limit_store[ip]) >= config.RATE_LIMIT_REQUESTS:
        return False

    rate_limit_store[ip].append(now)
    return True


def validate_input(question: str) -> str | None:
    """校验输入合法性，返回 None 表示通过，否则返回错误信息。"""
    if not question or not question.strip():
        return "问题不能为空"
    stripped = question.strip()
    if len(stripped) < config.MIN_INPUT_LENGTH:
        return f"问题太短，至少 {config.MIN_INPUT_LENGTH} 个字符"
    if len(stripped) > config.MAX_INPUT_LENGTH:
        return f"问题过长，最多 {config.MAX_INPUT_LENGTH} 个字符（当前 {len(stripped)} 字符）"
    # 检查是否全是空白/特殊字符
    if len(stripped.replace(" ", "").replace("\n", "")) == 0:
        return "问题不能仅为空白字符"
    return None


async def verify_auth(request: Request) -> None:
    """API Key 认证（可选）。"""
    if not config.AUTH_ENABLED:
        return
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="缺少 API Key（请通过 X-API-Key 头传递）")
    if api_key not in config.API_KEYS:
        raise HTTPException(status_code=403, detail="无效的 API Key")


# ================================================================
# Step 1: 环境与配置
# ================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEEPSEEK_API_KEY"] = config.DEEPSEEK_API_KEY

# ================================================================
# Step 2: 初始化 FastAPI
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    print("=" * 60)
    print("🚀 医学RAG问答系统启动中 (Agent+Skill 架构)...")
    print("=" * 60)
    print(f"🔒 认证: {'已启用' if config.AUTH_ENABLED else '未启用（可选）'}")
    print(f"⏱  限流: {'已启用' if config.RATE_LIMIT_ENABLED else '未启用'} ({config.RATE_LIMIT_REQUESTS}次/分钟)")
    print(f"📏 输入限制: {config.MIN_INPUT_LENGTH}~{config.MAX_INPUT_LENGTH} 字符")
    print(f"🌐 CORS 允许来源: {config.ALLOWED_ORIGINS}")
    print("-" * 60)
    yield
    print("🛑 服务已停止")


app = FastAPI(
    title="医学RAG问答系统 (Agent+Skill架构)",
    description="基于 RAG 架构的医疗问答系统，支持 Agent 工作流全链路可视化",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------- CORS 安全配置 ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ================================================================
# Step 3: 初始化服务
# ================================================================

# WSL兼容：将数据库文件同步到Linux原生路径
project_dir = os.path.dirname(os.path.abspath(__file__))
for db_name in ["milvus_agent.db", "pdf_agent.db"]:
    src = os.path.join(project_dir, db_name)
    dst = f"/tmp/{db_name}"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"📦 数据库已同步到Linux原生路径: {dst}")

# 嵌入模型
client_embedding = ZhipuAI(api_key=config.ZHIPU_API_KEY)
embedding_model = ZhipuAIEmbeddings(client_embedding)
print("✅ 智谱 Embedding-3 模型加载完成")

# Milvus 客户端
qa_client = MilvusClient(uri=config.MILVUS_QA_URI)
print(f"✅ QA向量库连接完成: {config.MILVUS_QA_URI}")

pdf_client = MilvusClient(uri=config.MILVUS_PDF_URI)
print(f"✅ PDF向量库连接完成: {config.MILVUS_PDF_URI}")

# LLM 模型
client_llm = create_deepseek_client()
print("✅ DeepSeek-Chat 模型创建成功")

# Skills & Agent
hybrid_search_skill = HybridSearchSkill(qa_client, embedding_model, collection_name="LangChainCollection")
parent_doc_skill = ParentDocSearchSkill(pdf_client, embedding_model, collection_name="LangChainCollection")
answer_gen_skill = AnswerGenerationSkill(client_llm)

medical_agent = MedicalAgent(
    hybrid_search_skill=hybrid_search_skill,
    parent_doc_skill=parent_doc_skill,
    answer_gen_skill=answer_gen_skill,
)

print("✅ MedicalAgent 初始化完成 (LangGraph StateGraph)")
print(f"   └─ 注册 Skills: {list(medical_agent.get_skill_registry().keys())}")
print("=" * 60)


# ================================================================
# Step 4: 公共中间件（限流 + 输入校验）
# ================================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """全局中间件：trace_id 绑定 + 限流 + 访问日志 + 耗时统计。

    trace_id 优先取上游传入的 X-Trace-Id（便于跨服务串联），否则新建。
    绑定后，本次请求内所有 log() 都会自动带上同一个 trace_id。
    """
    # 跳过健康检查等公共端点
    if request.url.path in ["/api/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    ip = request.client.host if request.client else "unknown"

    if not check_rate_limit(ip):
        METRICS["rate_limited_total"] += 1
        log("rate_limited", level="WARNING", ip=ip, path=request.url.path)
        return JSONResponse(
            status_code=429,
            content={
                "error": "请求过于频繁，请稍后再试",
                "status": 429,
                "retry_after": config.RATE_LIMIT_WINDOW,
                "time": _now(),
            },
        )

    trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
    started = time.time()

    with bind_trace(trace_id):
        METRICS["requests_total"] += 1
        log("request_start", method=request.method, path=request.url.path, ip=ip)
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - 记录后继续抛给上层处理器
            METRICS["errors_total"] += 1
            log(
                "request_crash",
                level="ERROR",
                error=type(exc).__name__,
                message=str(exc)[:300],
            )
            raise

        elapsed = (time.time() - started) * 1000
        response.headers["X-Response-Time"] = f"{elapsed / 1000:.3f}s"
        response.headers["X-Trace-Id"] = trace_id
        log("request_end", status=response.status_code, dur_ms=round(elapsed, 2))
        return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """业务异常统一出口：带上 code 与 trace_id，前端能定位。"""
    METRICS["errors_total"] += 1
    log("app_error", level="ERROR", code=exc.code, message=exc.message, path=request.url.path)
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": exc.message,
            "code": exc.code,
            "status": exc.http_status,
            "trace_id": current_trace_id(),
            "time": _now(),
        },
    )


# ================================================================
# Step 5: 请求与响应模型
# ================================================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=config.MIN_INPUT_LENGTH,
        max_length=config.MAX_INPUT_LENGTH,
        description="用户的医学问题",
    )


class ErrorResponse(BaseModel):
    error: str
    status: int
    time: str


# ================================================================
# Step 6: API 端点
# ================================================================

@app.post(
    "/",
    summary="基础问答",
    description="向后兼容的问答接口，前端使用",
    responses={
        200: {"description": "成功"},
        400: {"description": "输入校验失败"},
        401: {"description": "认证失败"},
        429: {"description": "请求过于频繁"},
        500: {"description": "服务内部错误"},
    },
)
async def chatbot(request: Request):
    """
    基础问答接口（向后兼容前端）

    Request:  {"question": "什么是糖尿病？"}
    Response: {"response": "...", "status": 200, "time": "..."}
    """
    # 认证
    if config.AUTH_ENABLED:
        await verify_auth(request)

    # 解析请求体
    try:
        json_post_raw = await request.json()
    except Exception:
        return {"response": "请求格式错误，请发送JSON", "status": 400,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    query = json_post_raw.get("question", "").strip()

    # 输入校验
    err = validate_input(query)
    if err:
        return {"response": err, "status": 400,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    try:
        # 设置超时
        result = await asyncio.wait_for(
            asyncio.to_thread(medical_agent.run, query),
            timeout=config.LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {"response": "请求超时，请简化问题后重试", "status": 504,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        return {"response": f"处理请求时出错: {str(e)}", "status": 500,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    if result.get("emergency"):
        METRICS["emergency_total"] += 1
    if result.get("degraded"):
        METRICS["degraded_total"] += 1

    return {
        "response": result["answer"],
        "status": 200,
        "time": _now(),
        "citations": result.get("citations", []),
        "degraded": result.get("degraded", False),
        "trace_id": current_trace_id(),
    }


@app.post(
    "/api/chat",
    summary="Agent 思考链问答",
    description="返回Agent完整思考链路，适合前端展示",
    responses={
        200: {"description": "成功"},
        400: {"description": "输入校验失败"},
        401: {"description": "认证失败"},
        429: {"description": "请求过于频繁"},
        500: {"description": "服务内部错误"},
    },
)
async def agent_chat(request: ChatRequest, req: Request):
    """
    Agent 思考链可视化问答接口。

    Request:  {"question": "糖尿病的饮食禁忌有哪些？"}
    Response: {
        "answer": "...",
        "thinking_steps": [...],
        "question_type": "diet",
        "retrieval_strategy": "hybrid",
        "skills_used": [...],
        "status": 200,
        "time": "..."
    }
    """
    # 认证
    if config.AUTH_ENABLED:
        await verify_auth(req)

    query = request.question.strip()

    # 额外输入校验
    err = validate_input(query)
    if err:
        raise HTTPException(status_code=400, detail=err)

    try:
        # 带超时的Agent调用
        start_time = time.time()
        result = await asyncio.wait_for(
            asyncio.to_thread(medical_agent.run, query),
            timeout=config.LLM_TIMEOUT,
        )
        elapsed = time.time() - start_time

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"请求超时（超过 {config.LLM_TIMEOUT} 秒），请简化问题后重试",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

    if result.get("emergency"):
        METRICS["emergency_total"] += 1
    if result.get("degraded"):
        METRICS["degraded_total"] += 1

    return {
        "answer": result["answer"],
        "thinking_steps": result.get("thinking_steps", []),
        "question_type": result.get("question_type", "general"),
        "retrieval_strategy": result.get("retrieval_strategy", "hybrid"),
        "context_length": result.get("context_length", 0),
        "citations": result.get("citations", []),
        "quality": result.get("quality", {}),
        "retry_count": result.get("retry_count", 0),
        "emergency": result.get("emergency", False),
        "degraded": result.get("degraded", False),
        "skills_used": list(medical_agent.get_skill_registry().keys()),
        "elapsed_seconds": round(elapsed, 2),
        "trace_id": current_trace_id(),
        "status": 200,
        "time": _now(),
    }


@app.get(
    "/api/skills",
    summary="查看 Skills 列表",
    description="返回所有已注册的 Skill 模块信息",
)
async def list_skills():
    """查看所有注册的 Skills 及其参数定义"""
    return {
        "skills": medical_agent.get_skill_registry(),
        "total": len(medical_agent.get_skill_registry()),
    }


@app.get(
    "/metrics",
    summary="进程内指标",
    description="请求量、错误量、降级与急症命中次数（公开端点，供本地排查）",
)
async def metrics():
    """轻量指标端点。

    注意：数据在进程内存里，多 worker 部署时需要换成 Prometheus 客户端，
    否则看到的是单个 worker 的数字。
    """
    return {
        "metrics": dict(METRICS),
        "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
        "auth_enabled": config.AUTH_ENABLED,
        "server_time": _now(),
    }


@app.get(
    "/api/health",
    summary="健康检查",
    description="服务健康状态检查（无需认证）",
)
async def health_check():
    """健康检查（公开端点，不受限流和认证影响）"""
    return {
        "status": "healthy",
        "architecture": "Agent + Skill (LangGraph)",
        "skills": list(medical_agent.get_skill_registry().keys()),
        "version": "2.1.0",
        "auth_enabled": config.AUTH_ENABLED,
        "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ================================================================
# Step 7: 启动服务
# ================================================================
if __name__ == "__main__":
    # 启动即自检：缺 Key、策略名写错这类问题，留到第一次请求才暴露时
    # 表现是"服务起来了但一问就 500"，排查成本远高于启动时直接失败。
    _problems = config.validate()
    if _problems:
        print("\n[FATAL] 配置有误，服务未启动：")
        for _p in _problems:
            print(f"  - {_p}")
        print("\n请参考 .env.example 补全配置后重试。\n")
        sys.exit(1)
    for _w in config.warnings():
        print(f"[WARN] {_w}")

    print(f"\n🌐 API 文档: http://localhost:{config.PORT}/docs")
    print(f"🔍 Skills 列表: http://localhost:{config.PORT}/api/skills")
    print(f"💬 Agent 问答: POST http://localhost:{config.PORT}/api/chat")
    print(f"📋 健康检查: http://localhost:{config.PORT}/api/health")
    print()
    uvicorn.run(app, host=config.HOST, port=config.PORT, workers=1)
