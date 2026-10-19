"""Embedding 抽象层。

两种实现，用途完全不同：

- **HashingEmbedder**：字符 n-gram 哈希到固定维度，零依赖、零下载、确定性。
  它捕捉的是"词面部分重合"，能力接近 TF-IDF。存在的意义是让评测能在 CI 里跑通，
  以及在没网/没模型的机器上仍然可复现。**不要**拿它的分数当语义效果。
- **BgeEmbedder**：sentence-transformers + BAAI/bge-small-zh-v1.5，真实中文语义向量。
  这是能拿来下结论的那个。

为什么不直接用生产用的智谱 embedding-3：它要 API Key，评测就该能在没有任何
外部依赖的情况下重跑。换 embedding 会改变绝对分数，但不改方法本身。
"""

from __future__ import annotations

import zlib
from typing import List, Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 (n, dim) 的 L2 归一化矩阵。"""
        ...


class HashingEmbedder:
    """字符 n-gram + 特征哈希。

    用 zlib.crc32 而不是内置 hash()：Python 的 hash() 对 str 带随机盐，
    同一条文本在不同进程里哈希值不同，评测就不可复现了。这个坑很隐蔽 ——
    单次运行看不出问题，跑两次对比才发现分数对不上。
    """

    name = "hashing"

    def __init__(self, dim: int = 4096, ngrams: tuple[int, ...] = (1, 2)) -> None:
        self.dim = dim
        self.ngrams = ngrams

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for n in self.ngrams:
            for i in range(len(text) - n + 1):
                gram = text[i : i + n]
                idx = zlib.crc32(gram.encode("utf-8")) % self.dim
                v[idx] += 1.0
        return v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class BgeEmbedder:
    """BAAI/bge-small-zh-v1.5，用 transformers 直接加载。

    为什么不用 sentence-transformers 的 SentenceTransformer 一行搞定：
    它在 6.1 + transformers 5.20 这套新版本上，读不到老模型仓库里的
    `config_sentence_transformers.json` 就直接抛 FileNotFoundError——
    而这个文件在 hub 上是存在的，是客户端只下了一部分。换版本、改缓存都是绕远路。

    BGE 的池化方式是取 **[CLS]** 而不是 mean pooling，实现就三行，
    自己写反而少一层版本兼容风险，也少一个重依赖。

    模型路径按顺序找：
    1. 环境变量 BGE_MODEL_PATH
    2. ~/.cache/bge-small-zh-v1.5（离线条目，手动下载的）
    3. hub 名（联网时自动下载）
    """

    name = "bge-small-zh"
    _BATCH = 32

    def __init__(self, model_path: str | None = None) -> None:
        import os
        from pathlib import Path

        import torch  # noqa: F401  # 提前失败，别等到 encode 才发现没装
        from transformers import AutoModel, AutoTokenizer

        candidates = [
            model_path,
            os.environ.get("BGE_MODEL_PATH"),
            str(Path.home() / ".cache" / "bge-small-zh-v1.5"),
            "BAAI/bge-small-zh-v1.5",
        ]
        resolved = next(
            (c for c in candidates if c and (not os.path.isdir(c) or os.path.exists(f"{c}/config.json"))),
            "BAAI/bge-small-zh-v1.5",
        )

        self._tok = AutoTokenizer.from_pretrained(resolved)
        self._model = AutoModel.from_pretrained(resolved)
        self._model.eval()
        self.dim = int(self._model.config.hidden_size)
        # 报告会写进仓库，不能把本机绝对路径（含用户名）带进去
        self.source = "local-dir" if os.path.isdir(str(resolved)) else str(resolved)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.dim), np.float32)

        chunks = []
        for i in range(0, len(texts), self._BATCH):
            batch = list(texts[i : i + self._BATCH])
            enc = self._tok(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._model(**enc)
            # BGE 取 [CLS] 位
            vec = out.last_hidden_state[:, 0]
            vec = torch.nn.functional.normalize(vec, dim=-1)
            chunks.append(vec.cpu().numpy().astype(np.float32))

        return np.vstack(chunks)


def get_embedder(name: str) -> Embedder:
    """按名字取 embedding。bge 不可用时**显式报错**，不静默降级 ——

    静默降级会让"语义检索"的结果悄悄变成词面匹配，报告里却不体现，
    这比跑不起来更糟。
    """
    if name in ("hashing", "hash"):
        return HashingEmbedder()
    if name in ("bge", "bge-small-zh"):
        return BgeEmbedder()
    raise ValueError(f"未知的 embedder: {name}（可选 hashing / bge）")
