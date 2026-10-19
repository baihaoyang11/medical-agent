"""
集中配置管理 — 所有 API Key、路径、模型参数统一管理

使用方式：
    from config import config
    api_key = config.ZHIPU_API_KEY

环境变量可覆盖默认值：
    export ZHIPU_API_KEY=xxx
"""

import os


class Config:
    # ===== API Keys（请通过环境变量设置）=====
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")

    # ===== 模型配置 =====
    EMBEDDING_MODEL: str = "embedding-3"        # 智谱 Embedding-3
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    # 医疗问答要可复现、少发挥，默认低温（旧版 0.7 偏高）
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # ===== 重试策略 =====
    UPSTREAM_RETRIES: int = int(os.getenv("UPSTREAM_RETRIES", "2"))
    RETRY_BASE_DELAY: float = float(os.getenv("RETRY_BASE_DELAY", "0.4"))
    MAX_SECONDARY_SEARCH: int = int(os.getenv("MAX_SECONDARY_SEARCH", "2"))

    # ===== 向量数据库路径（WSL需用Linux原生路径，不能用/mnt/d/）=====
    MILVUS_QA_URI: str = os.getenv("MILVUS_QA_URI", "/tmp/milvus_agent.db")
    MILVUS_PDF_URI: str = os.getenv("MILVUS_PDF_URI", "/tmp/pdf_agent.db")

    # ===== 检索参数 =====
    HYBRID_SEARCH_K: int = 10                   # 混合检索返回条数
    RRF_K: int = 100                            # RRF 融合参数
    CHILD_CHUNK_SIZE: int = 200                  # 子文档大小
    CHILD_CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 1000               # 父文档大小
    PARENT_CHUNK_OVERLAP: int = 200

    # ===== 重排（Rerank）=====
    # noop / keyword / cross_encoder
    # cross_encoder 需要 sentence-transformers + 本地模型，缺失时自动降级到 keyword
    RERANK_STRATEGY: str = os.getenv("RERANK_STRATEGY", "keyword")
    RERANK_MODEL_PATH: str = os.getenv("RERANK_MODEL_PATH", "")
    RERANK_WEIGHT: float = float(os.getenv("RERANK_WEIGHT", "0.6"))

    # ===== 离线入库（vectors1.py）=====
    EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # ===== 服务配置 =====
    HOST: str = "0.0.0.0"
    PORT: int = 8103

    # ===== PDF 数据路径 =====
    PDF_OUTPUT_DIR: str = "./pdf_output"
    PDF_DETAILED_TEXT: str = "./pdf_output/pdf_detailed_text.xlsx"
    DATA_JSONL: str = "./data/data.jsonl"

    # ===== 安全配置 =====
    # CORS: 允许的来源域名，生产环境请改为具体域名
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:8103"
    ).split(",")

    # 认证：API Key（为空则不启用认证）
    API_KEYS: list = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() == "true"

    # 请求限流（单IP）
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))   # 每分钟允许的请求数
    RATE_LIMIT_WINDOW: int = 60  # 窗口大小（秒）

    # 输入限制
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "2000"))
    MIN_INPUT_LENGTH: int = 2

    # 请求超时
    LLM_TIMEOUT: int = 60      # LLM调用超时（秒）
    SEARCH_TIMEOUT: int = 30   # 检索超时（秒）

    # ==================================================================
    # 启动自检
    # ==================================================================
    RERANK_STRATEGIES = ("noop", "keyword", "cross_encoder")

    @classmethod
    def validate(cls) -> list[str]:
        """返回致命配置问题列表（空列表表示可以启动）。

        为什么要在启动时做：缺 Key、策略名写错这类问题，如果留到第一次请求
        才暴露，表现是"服务起来了但一问就 500"，排查成本远高于启动即失败。
        这是 fail-fast 最便宜的落地方式。

        注意：不 import core.*，避免配置层反向依赖业务层。
        """
        problems: list[str] = []

        if not cls.ZHIPU_API_KEY:
            problems.append(
                "ZHIPU_API_KEY 未设置 —— embedding 无法调用。"
                "请在 .env 或环境变量中填入，见 .env.example"
            )
        if not cls.DEEPSEEK_API_KEY:
            problems.append(
                "DEEPSEEK_API_KEY 未设置 —— 答案生成无法调用。"
                "请在 .env 或环境变量中填入，见 .env.example"
            )

        if cls.AUTH_ENABLED and not cls.API_KEYS:
            problems.append(
                "AUTH_ENABLED=true 但 API_KEYS 为空 —— 所有请求都会被拒。"
                "要么填 API_KEYS，要么把 AUTH_ENABLED 设为 false"
            )

        if cls.RERANK_STRATEGY not in cls.RERANK_STRATEGIES:
            problems.append(
                f"RERANK_STRATEGY={cls.RERANK_STRATEGY!r} 不是合法值，"
                f"可选：{', '.join(cls.RERANK_STRATEGIES)}"
            )
        if cls.RERANK_STRATEGY == "cross_encoder" and not cls.RERANK_MODEL_PATH:
            problems.append(
                "RERANK_STRATEGY=cross_encoder 但 RERANK_MODEL_PATH 为空。"
                "填模型路径，或改用 keyword 策略"
            )

        if not 0.0 <= cls.LLM_TEMPERATURE <= 2.0:
            problems.append(f"LLM_TEMPERATURE={cls.LLM_TEMPERATURE} 超出 0~2 范围")

        if cls.EMBED_BATCH_SIZE < 1 or cls.EMBED_BATCH_SIZE > 64:
            problems.append(
                f"EMBED_BATCH_SIZE={cls.EMBED_BATCH_SIZE} 不合理，应为 1~64"
                "（智谱 embedding-3 单次上限 64 条）"
            )

        return problems

    @classmethod
    def warnings(cls) -> list[str]:
        """不致命、但值得在启动日志里提示的配置问题。"""
        notes: list[str] = []
        if not cls.AUTH_ENABLED:
            notes.append("AUTH_ENABLED=false：接口无认证，仅适合本地开发")
        if cls.MILVUS_QA_URI.startswith("/mnt/"):
            notes.append(
                f"MILVUS_QA_URI={cls.MILVUS_QA_URI} 位于 /mnt/ 下，"
                "Milvus Lite 在跨文件系统路径上容易加锁失败，建议改到 /tmp 或项目内"
            )
        if cls.LLM_TEMPERATURE > 0.5:
            notes.append(
                f"LLM_TEMPERATURE={cls.LLM_TEMPERATURE} 偏高，医疗问答建议 0~0.2（影响可复现性）"
            )
        return notes


# 全局单例
config = Config()
