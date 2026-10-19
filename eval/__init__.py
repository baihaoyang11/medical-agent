"""离线检索评测。

不依赖任何 API Key，可完全离线复现：

    python -m eval.run_eval --embedder hashing   # 零依赖，几秒
    python -m eval.run_eval --embedder bge       # 真实语义向量
"""
