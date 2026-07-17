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
    LLM_MODEL: str = "deepseek-chat"            # DeepSeek Chat
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_TEMPERATURE: float = 0.7

    # ===== 向量数据库路径（WSL需用Linux原生路径，不能用/mnt/d/）=====
    MILVUS_QA_URI: str = "/tmp/milvus_agent.db"
    MILVUS_PDF_URI: str = "/tmp/pdf_agent.db"

    # ===== 检索参数 =====
    HYBRID_SEARCH_K: int = 10                   # 混合检索返回条数
    RRF_K: int = 100                            # RRF 融合参数
    CHILD_CHUNK_SIZE: int = 200                  # 子文档大小
    CHILD_CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 1000               # 父文档大小
    PARENT_CHUNK_OVERLAP: int = 200

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


# 全局单例
config = Config()
