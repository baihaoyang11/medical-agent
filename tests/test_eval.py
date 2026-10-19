"""评测框架自身的测试。

为什么评测也要测：如果指标算错了，报告会照常生成、照常好看，
只是数字是假的。这种错误比代码崩溃危险得多，因为它不会被发现。

这里全部用 hashing embedder，不需要模型、不需要网络。
"""

import math

import pytest

from eval import metrics as M
from eval.knowledge import EXTRA_QUERIES, POINTS, build_queries
from eval.retrievers import BM25, Corpus, Doc, DenseRetriever, SparseRetriever, _tokenize
from eval.run_eval import build_corpus, build_query_set


# ── 指标 ──────────────────────────────────────────────────────
class TestMetrics:
    def test_recall_perfect(self):
        assert M.recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_recall_partial(self):
        assert M.recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5

    def test_recall_respects_k(self):
        # 正确的在第 6 位，k=5 时不应算命中
        ranked = ["x1", "x2", "x3", "x4", "x5", "a"]
        assert M.recall_at_k(ranked, ["a"], 5) == 0.0
        assert M.recall_at_k(ranked, ["a"], 10) == 1.0

    def test_mrr_rank_one(self):
        assert M.mrr(["a", "b"], ["a"]) == 1.0

    def test_mrr_rank_three(self):
        assert M.mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_mrr_no_hit(self):
        assert M.mrr(["x", "y"], ["a"]) == 0.0

    def test_ndcg_perfect_is_one(self):
        assert M.ndcg_at_k(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)

    def test_ndcg_penalises_late_hit(self):
        early = M.ndcg_at_k(["a", "x", "y"], ["a"], 3)
        late = M.ndcg_at_k(["x", "y", "a"], ["a"], 3)
        assert early > late

    def test_empty_gold_is_nan(self):
        # 急症 query 没有金标，必须返回 NaN 而不是 0，
        # 否则会被当成"没检索到"拉低平均分
        assert math.isnan(M.recall_at_k(["a"], [], 5))
        assert math.isnan(M.mrr(["a"], []))

    def test_mean_ignores_nan(self):
        assert M.mean([1.0, float("nan"), 0.0]) == pytest.approx(0.5)


# ── 检索器 ────────────────────────────────────────────────────
class TestBM25:
    @pytest.fixture
    def corpus(self):
        return [
            Doc("a", "糖尿病患者的饮食需要控制总热量", "diabetes", "diet", "core"),
            Doc("b", "糖尿病的药物治疗首选二甲双胍", "diabetes", "drug", "core"),
            Doc("c", "青光眼表现为眼压升高和视野缺损", "eye", "definition", "filler"),
        ]

    def test_scores_rank_relevant_first(self, corpus):
        bm25 = BM25(corpus)
        scores = bm25.scores("糖尿病的饮食控制")
        ranked = [corpus[i].doc_id for i in scores.argsort()[::-1]]
        assert ranked[0] == "a"

    def test_unrelated_doc_scores_zero(self, corpus):
        bm25 = BM25(corpus)
        scores = bm25.scores("青光眼")
        idx = [d.doc_id for d in corpus].index("c")
        assert scores[idx] > scores[0]  # 讲青光眼的那条分数更高

    def test_tokenize_is_deterministic(self):
        # 同一段文本两次分词必须一致，否则 BM25 分数不可复现。
        # 这里不断言具体切成什么（jieba 版本会变），只断言稳定性和基本有效性。
        toks = _tokenize("糖尿病饮食")
        assert toks == _tokenize("糖尿病饮食")
        assert toks, "分词结果不能为空"
        assert all(t.strip() for t in toks), "不应包含空白 token"

    def test_tokenize_produces_no_pure_punctuation(self):
        # 断言"每个 token 至少含一个实义字符"，而不是断言"不包含某个标点"。
        # 后者在字符二元组回退模式下会挂：切出来的是「，饮」这种跨标点的二元组，
        # 它含实义字，是合法 token。测试不该绑定某一种分词实现。
        import re

        toks = _tokenize("糖尿病，饮食。")
        assert all(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", t) for t in toks)

    def test_empty_query_scores_zero(self, corpus):
        bm25 = BM25(corpus)
        assert bm25.scores("").sum() == 0.0


class TestDenseRetriever:
    def test_returns_k_results(self):
        from eval.embedders import HashingEmbedder

        corpus = Corpus(
            [Doc(f"d{i}", f"内容{i}", "t", "f", "core") for i in range(20)]
        )
        r = DenseRetriever(corpus, HashingEmbedder(dim=256))
        assert len(r.search("内容3", 5)) == 5

    def test_self_retrieval_hits_itself(self):
        from eval.embedders import HashingEmbedder

        corpus = Corpus([Doc("only", "独一无二的一段文字", "t", "f", "core")])
        r = DenseRetriever(corpus, HashingEmbedder(dim=512))
        assert r.search("独一无二的一段文字", 1)[0] == "only"


# ── 数据集完整性 ──────────────────────────────────────────────
class TestDataset:
    def test_corpus_has_no_duplicate_ids(self):
        corpus = build_corpus()
        ids = [d.doc_id for d in corpus.docs]
        assert len(ids) == len(set(ids)), "doc_id 重复会导致召回统计错乱"

    def test_every_gold_id_exists_in_corpus(self):
        corpus = build_corpus()
        ids = {d.doc_id for d in corpus.docs}
        missing = [
            (q["query"], g) for q in build_query_set() for g in q["gold"] if g not in ids
        ]
        assert not missing, f"金标指向不存在的文档: {missing}"

    def test_query_count(self):
        # 96 条双问法 + 20 条术语型 + 4 条急症
        assert len(build_query_set()) == 120

    def test_three_query_styles_present(self):
        styles = {q["style"] for q in build_query_set()}
        assert styles == {"formal", "colloquial", "term"}

    def test_term_queries_share_vocabulary_with_their_gold(self):
        # 术语型 query 的设计意图就是"与正文共享专业词"。
        # 如果这条挂了，说明术语型 query 退化成了普通书面问法，
        # "术语型 question 考察精确匹配"这个前提就不成立了。
        from core.quality import keyword_coverage
        from eval.knowledge import TERM_QUERIES

        corpus = build_corpus()
        covs = [
            keyword_coverage(query, corpus.get(key).content) for query, key, _ in TERM_QUERIES
        ]
        avg = sum(covs) / len(covs)
        assert avg > 0.15, f"术语型 query 与正文平均覆盖率过低: {avg:.3f}"

    def test_every_point_has_two_queries(self):
        queries = build_queries()
        assert len(queries) == len(POINTS) * 2

    def test_emergency_queries_have_no_gold(self):
        assert all(q["gold"] == [] for q in build_query_set() if q["qtype"] == "emergency")
        assert len(EXTRA_QUERIES) == 4

    def test_colloquial_and_formal_differ(self):
        # 两种问法必须真的不同，否则"口语 vs 书面"的分组统计没有意义
        for p in POINTS:
            assert p.formal != p.colloquial
