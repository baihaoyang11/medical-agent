# 医学 RAG 问答系统（生产化改造版）

基于 **LangGraph + 双路混合检索 + 父子文档** 的医学问答服务。原始版本是毕业设计，
现在这份仓库是在原版基础上按生产标准补完的工程版本：真实分流的路由、
可回退的检索闭环、可追溯的引用、统一异常与可观测性、容器化与测试覆盖。

> 一句话概括改动：**原来只有"能跑通"，现在能回答"出了问题在哪、错在哪、怎么退"。**

---

## 1. 架构

```
                       ┌──────────────┐
         用户问题 ─────►│   router     │  规则分类 + 急症识别
                       └──────┬───────┘
                    急症? ┌───┴───┐ 否
                     ┌────▼──┐  ┌─▼──────────┐
                     │safety │  │  retrieve  │  HybridSearch / ParentDoc
                     └───┬───┘  └─┬──────────┘  按策略取 qa_k / pdf_k
                         │      ┌─▼──────────┐
                         │      │   verify   │  三维质量门（条数/长度/覆盖）
                         │      └─┬──────────┘
                         │   不达标 │ 且重试未超限
                         │      ┌─▼──────────┐
                         │      │   retry    │  计数 +1 并改写 query
                         │      └─┬──────────┘
                         │        └──────────► 回到 retrieve
                         │      ┌─▼──────────┐
                         │      │  generate  │  带引用编号生成 + 免责声明
                         │      └─┬──────────┘
                         ▼        ▼
                        END      END
```

### 为什么是图而不是一条 if-else 链

`verify → retry → retrieve` 是一个**有环**结构，用函数顺序调用表达会变成
手写 while 循环 + 局部变量堆积，状态散落各处。LangGraph 的 StateGraph 把
"状态"和"转移"分开：节点只负责读 State、返回增量，转移由条件边决定。
好处是重试次数、已用 query、降级标记这些都能通过 `checkpointer` 存档，
按 `thread_id` 恢复——这是后面做多轮和人工介入（HITL）的前提。

---

## 2. 目录结构

```
medical-agent/
├── agent.py                 # FastAPI 服务：中间件、异常出口、/metrics
├── agents/medical_agent.py  # LangGraph 编排器（5 节点 + 2 条条件边）
├── skills/                  # 能力层，只做一件事
│   ├── hybrid_search.py     #   稠密 + BM25 + RRF 融合
│   ├── parent_doc.py        #   父子文档召回
│   └── answer_gen.py        #   LLM 生成
├── core/                    # 与业务无关的横切能力
│   ├── routing.py           #   问题分类 → 检索策略表
│   ├── quality.py           #   检索质量三维判定
│   ├── rewrite.py           #   二次检索 query 改写
│   ├── rerank.py            #   可插拔重排（noop/keyword/cross_encoder）
│   ├── citations.py         #   引用构造与 prompt 渲染
│   ├── retry.py             #   指数退避（sleep 可注入，便于测试）
│   ├── errors.py            #   异常体系：code / http_status / retryable
│   ├── observability.py     #   trace_id + JSON Lines 日志 + span
│   ├── state.py             #   AgentState 定义
│   └── rrf.py               #   RRF 融合
├── vectors1.py              # 离线入库（批量 embedding + 断点可重跑）
├── preprocess.py            # PDF 结构化预处理
├── tests/                   # 51 个用例，不需要任何 API Key
├── Dockerfile / docker-compose.yml
└── .env.example
```

---

## 3. 快速开始

```bash
cp .env.example .env          # 填 ZHIPU_API_KEY / DEEPSEEK_API_KEY
pip install -r requirements.txt
python vectors1.py            # 构建向量库（首次）
python agent.py               # http://localhost:8103/docs
pytest tests/ -q              # 51 passed
```

容器方式：

```bash
docker compose up --build
curl http://localhost:8103/api/health
```

> Milvus 本地文件模式（Lite）**不能多实例共享同一个 .db**。生产部署请换成
> Milvus Standalone 服务地址，或保证每个副本独立库文件。

---

## 4. 关键设计决策

### 4.1 路由不是装饰品

原版 `_select_strategy()` 有六个分支，六个分支都返回 `"hybrid"`——路由存在但从不生效。
现在 `core/routing.py` 是一张策略表：

| 问题类型 | 策略 | qa_k | pdf_k | 上下文预算 |
|---|---|---|---|---|
| drug / diet | `hybrid_qa_heavy` | 15 | 1 | 4000 |
| disease_definition / clinical | `hybrid_pdf_heavy` | 4 | 4 | 6000 |
| diagnosis / prevention | `hybrid_wide` | 12 | 3 | 6000 |
| general | `hybrid` | 10 | 3 | 5000 |
| emergency | `qa_only` | 12 | 0 | 3000 |

`tests/test_routing.py::test_strategies_are_actually_different` 断言策略参数至少有
4 种不同组合——**防止再次退化成"路由空转"**。这类回归靠人眼是看不出来的。

### 4.2 质量判定换掉"字符数 > 100"

旧判定只看上下文长度，一段完全无关的长文也能通过。现在三个信号：

- `doc_count` —— 有没有召回
- `context_chars` —— 够不够支撑一段回答
- `keyword_coverage` —— query 的字符二元组有多少出现在上下文里

覆盖率用**字符二元组**而不是分词：中文医学实体（"非小细胞肺癌"）分词容易切碎，
二元组对分词器零依赖，换模型不影响判定逻辑。
`test_long_but_irrelevant_context_fails` 就是钉住这个漏洞的用例。

### 4.3 重试是"改写后重试"，不是"原样再试"

原样重试只会拿到同样的结果。`core/rewrite.py` 按轮次递进：
第 1 次去停用词（"请问…有哪些" → 实体词），第 2 次加同义词扩展。
`used_queries` 记录本轮用过哪些 query，结果按 `(doc_id, content[:60])` 去重，
避免第二轮把第一轮的相同段落又塞一遍。

### 4.4 重排可插拔、可降级

| 策略 | 依赖 | 用途 |
|---|---|---|
| `noop` | 无 | 消融对比（关掉重排看指标差异） |
| `keyword` | 无 | 默认。词面覆盖率 + 原排名分加权，CI 里无 GPU 也能验证链路 |
| `cross_encoder` | sentence-transformers + 本地模型 | 懒加载，装不上自动退回 keyword 并标记 `degraded` |

不做"重排后按阈值过滤"：分数阈值不可跨模型移植，卡阈值等于把模型版本焊死在业务代码里。

### 4.5 引用是可追溯的，不是装饰

`citations` 从 `[]` 变成 `{rank, doc_id, source, page, snippet}` 列表，
去重键是 `(doc_id, page)`，同时以 `[1] 生理学.pdf 第 12 页` 的形式注入 prompt，
让模型能按编号引用。用户点开引用能回到原文页——这是医疗场景的硬要求。

### 4.6 急症直接短路

`emergency`（胸痛/呼吸困难/昏迷/自杀倾向/服毒…）优先级高于所有分类，
命中后**不检索、不生成**，直接返回就医指引 + 免责声明。
理由：这类问题等一次 LLM 生成就是几十秒，且生成内容不可控。

### 4.7 降级矩阵

| 故障点 | 行为 | 用户看到 |
|---|---|---|
| HybridSearch 失败 | 记为"无 QA 上下文"，继续 ParentDoc | 正常答案，`degraded=true` |
| ParentDoc 失败 | 同上，只用 QA | 正常答案 |
| 两者都失败 | 上下文为空 → verify 不通过 → 重试 → 仍失败则生成兜底 | 提示信息不足并建议就医 |
| LLM 失败（重试耗尽） | 返回"检索到的原文 + 无法生成" | 原文片段，可自行阅读 |
| 重排模型不可用 | 退回 keyword 重排 | 体验无差异 |

---

## 5. API

### `POST /api/chat`（推荐）

```bash
curl -X POST http://localhost:8103/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"2型糖尿病患者的饮食禁忌有哪些？"}'
```

```json
{
  "answer": "……（末尾附免责声明）",
  "thinking_steps": [
    "[Router] 类型=diet 策略=hybrid_qa_heavy (qa_k=15, pdf_k=1)",
    "[Retrieve] HybridSearch 命中 15 条（hybrid_dense+bm25+rrf）",
    "[Retrieve] keyword 重排 15 条",
    "[Verify] 通过：15 条 / 4380 字符 / 覆盖 85%",
    "[Generate] 生成 620 字符，引用 5 处"
  ],
  "citations": [{"rank": 1, "doc_id": "…", "source": "生理学.pdf", "page": 12, "snippet": "…"}],
  "quality": {"passed": true, "coverage": 0.85, "context_chars": 4380, "doc_count": 15},
  "retry_count": 0,
  "emergency": false,
  "degraded": false,
  "trace_id": "6f1c…",
  "elapsed_seconds": 3.42
}
```

`thinking_steps` 不是日志，是**给排障和给面试官看的东西**：
"为什么这次慢"、"为什么答案里没有引用"、"重试了几轮"，看一眼就知道。

### 其它端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/` | 兼容旧前端的简化接口 |
| GET | `/api/health` | 健康检查（免限流免认证） |
| GET | `/api/skills` | Skills 清单与参数定义 |
| GET | `/metrics` | 请求量 / 错误 / 超时 / 限流 / 急症 / 降级 |

错误统一出口：

```json
{"error": {"code": "upstream_timeout", "message": "…", "retryable": true}, "trace_id": "6f1c…"}
```

---

## 6. 测试

```bash
pytest tests/ -q      # 51 passed，不需要任何 API Key
```

| 文件 | 覆盖内容 |
|---|---|
| `test_routing.py` | 分类、策略表、急症短路、**防"路由空转"回归** |
| `test_quality_rewrite.py` | 三维质量门、长而无关的上下文被拦、改写递进 |
| `test_retry.py` | 退避递增、`max_delay` 上限、不可重试异常不重试 |
| `test_citations.py` | 去重、截断、prompt 渲染、空输入 |
| `test_rerank.py` | 排序稳定性、脏输入、主路失败走备路 |
| `test_agent_graph.py` | 真实跑 LangGraph 图：急症短路 / 一次通过 / 改写重试 / 重试上限 / 去重 / 预算丢弃 |

集成测试用 `FakeSkill` + `FakeAnswer` 替掉外部依赖，所以能进 CI——
**"需要真 Key 才能测"等于"从来没人测"**。

---

## 7. 已知边界（没做的部分，以及为什么）

诚实列出来，比写成"已完成"更有用：

1. **限流在进程内存**：`RATE_LIMIT_REQUESTS` 用 dict 计数，多 worker 各自计，
   实际额度会翻倍。上生产换 Redis 或网关限流。
2. **指标未接 Prometheus**：`/metrics` 返回进程内数字，重启即清零，
   且多副本要自己汇总。接 `prometheus_client` 是下一步。
3. **BM25 索引全量驻留内存**：十万级文档内存吃紧，且无法增量更新
   （新文档必须重启进程才进索引）。正解是换 Milvus 的 sparse 倒排或 Elasticsearch。
4. **无多轮上下文**：State 支持 `thread_id` 复用，但没有做"指代消解"
   （"它有什么副作用"里的"它"指什么）。
5. **无自动评测集**：只有单测和人工抽查，没有 recall@k / 答案忠实度的
   离线评测。要做的话先攒 100 条 query-answer 对当金标。
6. **`test.py` 是手工验收脚本**，不是自动化测试，跑起来要真 Key、要服务在跑。

---

## 8. 技术栈

Python 3.11 / FastAPI / LangGraph / Milvus(Lite) / 智谱 embedding-3(1024d) /
DeepSeek-Chat / rank-bm25 + jieba / Docker

---

## 9. 免责声明

本项目仅用于学习与研究，**不构成任何医疗建议**。所有回答末尾附带免责声明；
涉及急症的一律引导至线下就医。
