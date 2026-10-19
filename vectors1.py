import os
import sys
from pydantic import BaseModel
# --- 兼容性补丁：解决 Pydantic 2.x 路径问题 ---
# 强制将 pydantic.v1 映射到 langchain_core 期望的位置
try:
    from pydantic import v1 as pydantic_v1
    # 创建正确的模块映射
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
    print("✅ Pydantic 兼容性补丁已加载 (V2 -> V1 映射成功)")
except Exception as e:
    print(f"⚠️ 补丁加载失败: {e}")
from tqdm import tqdm
import json
import uuid
import time
from zhipuai import ZhipuAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
import pandas as pd
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

from config import config
from core.errors import UpstreamError
from core.retry import call_with_retry

zhipu_api_key = os.getenv('ZHIPU_API_KEY')
if not zhipu_api_key:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")
client = ZhipuAI(api_key=zhipu_api_key)

class ZhipuAIEmbeddings(Embeddings):
    """智谱 Embedding-3 封装。

    旧实现逐条调用 API：一万篇文档就是一万次 HTTPS 往返，按 200ms/次
    算要 33 分钟，且中途任何一次 429/5xx 都会让整批入库前功尽弃。

    这里改成：
    - 批量提交（一次 32 条），往返次数降到 1/32
    - 失败按指数退避重试，重试仍失败就抛出，由调用方决定是否落盘检查点
    - 按 response.data[i].index 回填顺序 —— 批量接口不保证返回顺序与入参一致，
      错位会让"向量 A 对应文档 B"，是最难排查的数据污染
    """

    def __init__(self, client, batch_size: int | None = None):
        self.client = client
        self.batch_size = batch_size or getattr(config, "EMBED_BATCH_SIZE", 32)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        def _call() -> list[list[float]]:
            try:
                response = self.client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=texts,
                )
            except Exception as exc:  # noqa: BLE001
                raise UpstreamError(f"embedding 批量调用失败: {exc}") from exc

            # 按 index 回填，不假设顺序
            ordered: list[tuple[int, list[float]]] = []
            for item in response.data:
                idx = getattr(item, "index", None)
                ordered.append((idx if idx is not None else len(ordered), item.embedding))
            ordered.sort(key=lambda pair: pair[0])
            return [vec for _, vec in ordered]

        return call_with_retry(
            _call,
            retries=config.UPSTREAM_RETRIES,
            base_delay=config.RETRY_BASE_DELAY,
            op_name="embed_batch",
        )

    def embed_documents(self, texts):
        texts = list(texts)
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            embeddings.extend(self._embed_batch(chunk))
            if len(texts) > self.batch_size:
                print(f"  已向量化 {start + len(chunk)}/{len(texts)} 条")
        return embeddings

    def embed_query(self, text):
        return self._embed_batch([text])[0]

    def embed_documents_single_batch(self, texts):
        """debug 用：确认批量接口在给定 batch_size 下不超限。"""
        return self._embed_batch(list(texts))

class Milvus_vector():
    def __init__(self, client, uri=None):
        self.URI = uri or config.MILVUS_QA_URI
        # 这里的 client 是你传入的 ZhipuAiClient 实例
        self.embeddings = ZhipuAIEmbeddings(client=client) 
        
        # 1. 定义稠密向量索引（语义搜索）：使用内积 (IP) 衡量相似度
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        # 2. 定义稀疏向量索引（关键词搜索）：使用 BM25 算法
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }

    def create_vector_store(self, docs):
        # 预取前10条用于初始化 Collection 结构
        init_docs = docs[:10]

        if os.path.exists(self.URI):
            os.remove(self.URI)

        # 初始化并创建 Milvus 存储实例
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(), 
            index_params=[self.dense_index, self.sparse_index], # 双索引：稠密 + 稀疏
            vector_field=["dense", "sparse"],
            connection_args={"uri": self.URI},
            consistency_level="Bounded", # 一致性级别：有界一致性
            drop_old=False, # 不删除旧数据，方便增量更新
        )
        
        print("\n✅ 已初始化创建 Milvus ‼")

        # 批量插入剩余数据
        count = 10
        temp = []
        # 从第11条开始遍历
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            # 每积累 5 条执行一次批量插入（Batch Insert）
            if len(temp) >= 5:
                # 注意：这里调用的是同步/异步添加方法，取决于你的环境
                self.vectorstore.add_documents(temp) 
                count += len(temp)
                temp = []
                # 适当休眠，防止 API 请求频率过高（Rate Limit）
                time.sleep(1) 
                print(f"已插入 {count} 条数据......")

        # 处理最后不足 5 条的余项
        if temp:
            self.vectorstore.add_documents(temp)
            count += len(temp)

        print(f"总共插入 {count} 条数据......")
        print("\n✅ 已创建 Milvus 索引完成 ‼")
        return self.vectorstore
class Pdf_retriever():
    def __init__(self, client, uri=None):
        self.URI = uri or config.MILVUS_PDF_URI
        self.embeddings = ZhipuAIEmbeddings(client=client)
        # 定义索引类型 [cite: 2570-2577]
        self.dense_index = {"metric_type": "IP", "index_type": "IVF_FLAT"}
        self.sparse_index = {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX"}
        
        # 核心：定义文档存储与分割器 [cite: 2578-2591]
        self.docstore = InMemoryStore()
        # 子分割器：用于向量搜索，粒度更细（200字）
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        # 父分割器：用于召回展示，保证上下文完整（1000字）
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def create_pdf_vector_store(self, docs):
        # 初始化双索引 Milvus 向量库 [cite: 2592-2609]
        self.milvus_vectorstore = Milvus(
            embedding_function=self.embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[self.dense_index, self.sparse_index],
            connection_args={"uri": self.URI},
            consistency_level="Bounded",
            drop_old=False
        )
        
        # 封装父子文档检索器 [cite: 2611-2617]
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.milvus_vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )
        
        # 批量入库
        count = 0
        temp = []
        for doc in tqdm(docs):
            temp.append(doc)
            if len(temp) >= 10:
                self.retriever.add_documents(temp)
                count += len(temp)
                temp = []
                time.sleep(1)
        # 旧版漏了这一步：不足 10 条的尾巴永远不会入库，
        # 而 PDF 页数通常不是 10 的整数倍，等于每本书丢几页
        if temp:
            self.retriever.add_documents(temp)
            count += len(temp)
            temp = []

        print(f"✅ PDF 入库完成：{count} 页")
        return self.retriever

def prepare_pdf_document(file_path="./pdf_output/pdf_detailed_text.xlsx"):
    """读取预处理好的 PDF Excel 文件 [cite: 2648-2662]"""
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['text_content'])
    documents = []
    for _, row in df.iterrows():
        text_content = str(row['text_content'])
        doc = Document(
            page_content=text_content.strip(),
            metadata={"doc_id": str(uuid.uuid4()), "source": row['file_name'], "page": row['page_number']}
        )
        documents.append(doc)
    print(f"成功加载 {len(documents)} 个 PDF 页面文档")
    return documents
def prepare_document(file_path=['./data/data.jsonl', './data/train.jsonl']):
    """
    逐条读取文本数据，封装为 Document 对象并添加唯一 ID
    """
    file_path1 = file_path[0]
    count = 0
    docs = []

    with open(file_path1, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 解析 JSON 每一行
            content = json.loads(line.strip())
            
            # 2. 拼接 Prompt：将问题和回答组合，形成完整的语义语境
            # 这种拼接方式有助于 Embedding 模型更好地理解上下文关系
            prompt = content['query'] + "\n" + content['response']

            # 3. 创建 Document 对象
            # metadata 里的 doc_id 是为了后续检索回溯，防止幻觉的关键标识
            temp_doc = Document(
                page_content=prompt, 
                metadata={"doc_id": str(uuid.uuid4())}
            )
            docs.append(temp_doc)
            count += 1

    print(f"✅ 已加载 {count} 条数据！")
    return docs

if __name__ == "__main__":
    # 第一路：加载原始 QA 问答对数据 [cite: 2667-2674]
    # qa_docs = prepare_document()
    # milvus_vectorstore = Milvus_vector(client=client)
    # milvus_vectorstore.create_vector_store(qa_docs)
    # print("✅ 第一路：QA 问答对入库完成")

    # 第二路：加载预处理后的 PDF 教材数据 [cite: 2676-2685]
    pdf_docs = prepare_pdf_document()
    pdf_retriever_instance = Pdf_retriever(client=client)
    pdf_retriever = pdf_retriever_instance.create_pdf_vector_store(pdf_docs)
    print("✅ 第二路：PDF 权威教材入库完成")

    print("\n🚀 全部初始化完成，系统已具备多路召回能力！")