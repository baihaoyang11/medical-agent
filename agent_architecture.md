# 🏗️ 医学RAG系统 — Agent+Skill 架构文档

## 架构概览

本项目从传统线性RAG管线升级为 **Agent + Skill 可插拔架构**，使用 **LangGraph** 编排多节点智能体协作流程。

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI 服务层                        │
│  POST / (兼容前端)  │  POST /api/chat (思考链)           │
│  GET /api/skills    │  GET /api/health                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              MedicalAgent (LangGraph)                     │
│                                                          │
│   START → [router] → [retrieve] → [verify] → [generate]  │
│              │            │           │           │       │
│              ▼            ▼           ▼           ▼       │
│        分析问题类型   调用Skills   质量检查    LLM生成     │
│        选择检索策略   收集上下文   决定重试    最终答案     │
│                                                          │
│  每个节点记录 thinking_steps（决策可追溯）                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Skills 技能层                          │
│                                                          │
│  ┌──────────────────┐ ┌──────────────────┐              │
│  │ HybridSearch     │ │ ParentDocSearch  │              │
│  │ QA库语义检索      │ │ PDF教材检索       │              │
│  │ MilvusClient API │ │ MilvusClient API │              │
│  └──────────────────┘ └──────────────────┘              │
│  ┌──────────────────┐                                    │
│  │ AnswerGeneration │                                     │
│  │ DeepSeek LLM     │                                    │
│  │ 医学提示词工程     │                                    │
│  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

## Agent 工作流详解

### 节点1: Router（路由）
- **职责**: 分析用户问题类型，决定最优检索策略
- **问题分类**: disease_definition / diet / drug / diagnosis / prevention / clinical / general
- **策略选择**: 
  - 疾病定义/临床指标 → hybrid（QA+PDF双路检索）
  - 鉴别诊断 → hybrid（多维度检索）
  - 饮食/预防/药物 → hybrid（QA+教材综合）

### 节点2: Retrieve（检索）
- **职责**: 调用 Skills 收集上下文
- **Skill 1 - HybridSearch**: 从QA问答对库中语义检索相关知识
- **Skill 2 - ParentDocSearch**: 从PDF医学教材库中检索权威段落
- **输出**: 拼接后的完整上下文

### 节点3: Verify（验证）
- **职责**: 检查检索结果质量
- **逻辑**: 上下文少于100字符时标记需要二次检索
- **扩展**: 可加入答案-原文一致性校验

### 节点4: Generate（生成）
- **职责**: 基于上下文生成最终答案
- **Skill - AnswerGeneration**: DeepSeek-Chat + 医学提示词
- **特点**: 优先使用检索信息，无信息时诚实告知

## Skills 注册表

| Skill | ID | 依赖 | 描述 |
|-------|-----|------|------|
| HybridSearch | `hybrid_search` | MilvusClient + ZhipuAI Embedding | 稠密向量语义检索，从QA库召回医学问答对 |
| ParentDocSearch | `parent_doc_search` | MilvusClient + ZhipuAI Embedding | 从PDF教材库检索权威医学知识 |
| AnswerGeneration | `answer_generation` | DeepSeek-Chat LLM | 基于检索上下文生成专业医学答案 |

## 数据流示例

```
用户: "糖尿病的饮食禁忌有哪些？"

1. [Router Agent] 分析: "糖尿病的饮食禁忌有哪些？..." 
   → 类型: diet, 策略: hybrid

2. [Retrieve Agent] 调用 HybridSearch Skill 
   → 从QA库召回 10 条文档

3. [Retrieve Agent] 调用 ParentDocSearch Skill 
   → 从PDF教材召回 3 条段落

4. [Retrieve Agent] 上下文聚合完成 
   → 总长度 1665 字符

5. [Verify Agent] 检索质量检查 
   → 通过 (上下文 1665 字符)

6. [Generate Agent] 调用 AnswerGeneration Skill 
   → DeepSeek-Chat 生成答案 (666 字符)

最终答案: "根据检索到的信息，关于糖尿病的饮食禁忌..."
```

## 代码结构

```
medical-agent/
├── agent.py               # FastAPI入口 + Agent/Skill初始化
├── config.py              # 集中配置管理 (API Key/路径/参数)
├── model1.py              # 模型层 (ZhipuAI Embedding + DeepSeek LLM)
├── core/
│   ├── base_skill.py      # Skill基类 (标准化输入/出接口)
│   └── state.py           # AgentState定义 (LangGraph状态)
├── skills/
│   ├── hybrid_search.py   # 混合检索Skill
│   ├── parent_doc.py      # PDF教材检索Skill
│   └── answer_gen.py      # 答案生成Skill
├── agents/
│   └── medical_agent.py   # LangGraph MedicalAgent (4节点图)
├── preprocess.py          # PDF预处理 (保持不变)
├── vectors1.py            # 向量库构建 (保持不变)
├── frontend/              # Web前端 (保持不变)
└── data/                  # 训练数据 (保持不变)
```

## 扩展到新 Skill

继承 `BaseSkill` 即可添加新能力：

```python
from core.base_skill import BaseSkill, SkillResult

class DrugInteractionSkill(BaseSkill):
    def __init__(self, drug_db_client):
        super().__init__()
        self._name = "drug_interaction_check"
        self._description = "检查药物相互作用，返回风险等级和建议"
        self._parameters = {
            "drug_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要检查的药物名称列表"
            }
        }

    def execute(self, drug_names=None, **kwargs):
        # ... 实现药物相互作用检查逻辑
        return SkillResult.ok(data=interactions)
```

## 面试亮点

1. **Agent自主决策**: Agent根据问题类型自动选择检索策略（不是硬编码if/else管线）
2. **Skill可插拔**: 每个能力独立封装，统一BaseSkill接口，新增能力无需改动Agent核心
3. **思考链可视化**: `/api/chat` 端点返回完整 thinking_steps，展示Agent决策过程
4. **双路检索融合**: QA问答对 + PDF权威教材，语义搜索 + 关键词匹配
5. **LangGraph工业级编排**: 使用LangGraph StateGraph构建有向无环工作流
6. **标准化Tool Schema**: 每个Skill导出OpenAI Function Calling兼容的工具定义
