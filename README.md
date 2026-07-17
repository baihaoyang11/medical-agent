# 🎓 医学RAG问答系统 - 毕业设计项目

基于 **检索增强生成 (RAG)** 技术的智能医学问答系统，采用 **双路混合检索 + 父子文档策略** 实现高质量的医学知识问答。

---

## 📁 项目结构

```
医学RAG/
├── agent.py              # ⭐ FastAPI服务主文件（核心业务逻辑）
├── model1.py             # 模型层（智谱Embedding + DeepSeek LLM）
├── vectors1.py           # 向量数据库层（Milvus双路召回系统）
├── preprocess.py         # PDF文档预处理
├── test.py               # ⭐ 综合测试脚本
├── data/                 # 训练数据目录
│   ├── data.jsonl        # QA问答对数据
│   ├── dialog.jsonl      # 对话数据
│   └── train.jsonl       # 训练集
├── pdf_documents/        # 原始PDF医学教材
├── pdf_output/           # PDF处理后的结构化数据
│   ├── pdf_detailed_text.xlsx
│   └── pdf_extraction_summary.xlsx
├── milvus_agent.db       # QA问答对向量数据库（Milvus）
└── pdf_agent.db          # PDF文档向量数据库（Milvus）
```

---

## 🏗️ 技术架构

### 核心技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| **嵌入模型** | 智谱AI Embedding-3 | 文本向量化（1024维） |
| **语言模型** | DeepSeek-Chat | 答案生成大模型 |
| **向量数据库** | Milvus (SQLite模式) | 本地向量存储与检索 |
| **Web框架** | FastAPI + Uvicorn | RESTful API服务 |
| **文档处理** | pdfplumber + pandas | PDF文本与表格提取 |

### 检索策略

```
用户问题
    ├── 路径1: QA问答对召回（Milvus双索引）
    │   ├── 稠密向量检索（语义搜索）
    │   ├── 稀疏向量检索（BM25关键词搜索）
    │   └── RRF算法融合排序
    │
    └── 路径2: PDF父文档召回（父子文档策略）
        ├── 子文档（200字）→ 向量检索
        └── 父文档（1000字）→ 上下文展示

↓

DeepSeek-Chat 生成最终答案
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install fastapi uvicorn
pip install pymilvus langchain-milvus
pip install zhipuai pdfplumber openai
pip install pandas tqdm requests

# 下载Milvus（如未安装）
# 本项目使用SQLite模式，无需单独安装Milvus服务
```

### 2. 数据准备

```bash
# 1. 放置医学PDF教材到 pdf_documents/ 目录
# 2. 运行PDF预处理
python preprocess.py

# 3. 构建向量数据库（如数据库文件不存在）
python vectors1.py
```

### 3. 启动服务

```bash
# 启动RAG问答服务
python agent.py

# 服务将在 http://0.0.0.0:8103 启动
```

### 4. 运行测试

```bash
# 确保服务已启动后，在另一个终端运行
python test.py
```

---

## 🧪 测试文件说明

`test.py` 包含以下6大测试模块：

### ✅ 测试1: 向量数据库连接
- 检查数据库文件完整性
- 测试Milvus连接
- 验证检索功能

### ✅ 测试2: 嵌入模型
- 智谱Embedding-3单文本测试
- 批量嵌入性能测试

### ✅ 测试3: 大语言模型
- DeepSeek-Chat模型调用测试
- 响应时间评估

### ✅ 测试4: API服务
- 服务健康检查
- API问答接口测试

### ✅ 测试5: 检索质量评估
- 稠密向量 vs 混合检索对比
- 多查询召回质量测试

### ✅ 测试6: 端到端RAG问答
- 完整双路召回流程
- 答案生成质量评估

---

## 📡 API接口

### 问答接口

**请求:**
```bash
POST http://localhost:8103/
Content-Type: application/json

{
  "question": "糖尿病患者的饮食建议是什么？"
}
```

**响应:**
```json
{
  "response": "根据医学资料，糖尿病患者应遵循以下饮食原则...",
  "status": 200,
  "time": "2026-05-09 20:43:06"
}
```

---

## 📊 项目特色

### 1️⃣ 双路混合检索
- **稠密向量检索**：理解语义（如"血糖"和"糖尿病"关联）
- **稀疏向量检索**：精确匹配关键词（如专业术语）
- **RRF融合算法**：智能排序，平衡语义与关键词

### 2️⃣ 父子文档策略
- **子文档**：200字小块，提高检索召回率
- **父文档**：1000字大块，提供完整上下文

### 3️⃣ 多数据源融合
- QA问答对（`data.jsonl`）
- 医学PDF教材（生物化学、生理学、微生物学）
- 动态上下文组合

### 4️⃣ 医学专业提示词
```python
SYSTEM_PROMPT = "System:你是一个非常得力的医学助手,你可以通过从数据库中检索出的信息找到问题的答案。"
```

---

## 🎯 使用示例

### Python调用示例

```python
import requests

# 1. 发送问题
response = requests.post(
    "http://localhost:8103/",
    json={"question": "什么是糖尿病？"},
    timeout=30
)

# 2. 获取答案
result = response.json()
print(result["response"])
```

### cURL调用示例

```bash
curl -X POST http://localhost:8103/ \
  -H "Content-Type: application/json" \
  -d '{"question": "糖尿病的饮食禁忌有哪些？"}'
```

---

## 📝 配置说明

### API密钥配置

在 `agent.py` 和 `model1.py` 中：

```python
# 智谱AI API Key（用于Embedding）
ZHIPU_API_KEY = "your_zhipu_api_key_here"

# DeepSeek API Key（用于LLM）
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
```

### 服务端口配置

```python
# 在 agent.py 最后一行
uvicorn.run(app, host='0.0.0.0', port=8103, workers=1)
```

---

## 🔧 故障排查

### 问题1: CUDA驱动版本过旧警告
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```
**解决方案**：忽略即可，本项目默认使用CPU运行

### 问题2: Pydantic兼容性警告
**解决方案**：已在代码中添加兼容性补丁（无需处理）

### 问题3: API连接失败
**解决方案**：检查API Key是否正确，网络是否正常

---

## 📈 后续优化方向

1. **性能优化**
   - [ ] 添加Redis缓存层
   - [ ] 实现批量查询接口
   - [ ] 向量数据库索引优化

2. **功能扩展**
   - [ ] 添加对话历史记忆
   - [ ] 实现答案来源溯源
   - [ ] 支持多轮对话

3. **部署优化**
   - [ ] Docker容器化
   - [ ] 负载均衡配置
   - [ ] 监控日志系统

---

## 👨‍🎓 作者信息

- **项目名称**: 医学RAG问答系统
- **项目类型**: 毕业设计
- **完成时间**: 2026年5月
- **技术栈**: Python + FastAPI + Milvus + DeepSeek

---

## 📄 许可证

本项目仅用于学习与研究目的。

---

**🎉 祝答辩顺利！**
