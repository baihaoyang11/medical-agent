# data/ —— 语料目录

这个目录**默认不随仓库提交**（版权 + 体积），所以克隆下来是空的。
下面说明每个文件是什么、从哪来、没有的话怎么跑通。

## 目录约定

```
data/
├── data.jsonl      # 生产语料：QA 问答对，一行一条
├── demo.jsonl      # 演示语料：脚本生成的 24 条，仅用于跑通链路
└── README.md       # 本文件
```

## data.jsonl 的格式

一行一个 JSON 对象，必须有 `query` 和 `response` 两个字符串字段：

```json
{"query": "糖尿病患者的饮食禁忌有哪些", "response": "糖尿病患者应控制总热量与碳水化合物比例……"}
{"query": "高血压需要终身服药吗", "response": "原发性高血压目前无法根治……"}
```

入库时代码会把两者拼成 `query + "\n" + response` 作为一段文本
（见 `vectors1.py:prepare_document`），这样向量里同时包含"问题语义"和"答案内容"，
用户用口语提问也能命中。

## 从哪来

按优先级：

1. **公开医学问答数据集**（推荐，注意核对许可证是否允许商用）
   - 中文医疗对话/问答类数据集在 Hugging Face 上有多个公开版本，搜索
     `medical qa chinese` 一类关键词，下载后转成上面的 JSONL 格式。
   - 许可证是 MIT / CC-BY 的可以放心用于个人项目；**许可证不明或仅限研究用途的，
     不要用在商业项目里**，这条在真实交付中会被法务卡住。
2. **院内/自建的专家问答**：真实项目里这是主体，需要有资质的医生审核。
   注意患者信息必须脱敏后才能入库。
3. **演示数据**：`python scripts/prepare_data.py --demo`，24 条，
   只用来验证链路通不通，**不能用来评估效果**。

## 没有真实语料时怎么开始

```bash
python scripts/prepare_data.py --demo   # 生成 data/demo.jsonl
python scripts/prepare_data.py --check  # 确认就绪
python vectors1.py                      # 建库（需要 ZHIPU_API_KEY）
```

## 两个容易踩的坑

1. **别把大文件提交进 git**。`.gitignore` 已经忽略 `data/*.jsonl`，
   真实语料请放在对象存储或单独的语料仓库，仓库里只留生成脚本和 schema 说明。
2. **PDF 那一路的数据不在这里**。它在 `pdf_output/pdf_detailed_text.xlsx`，
   由 `preprocess.py` 从 `pdf_documents/*.pdf` 抽取生成。两条链路的语料要分开维护，
   因为它们的更新频率和审核流程完全不同。
