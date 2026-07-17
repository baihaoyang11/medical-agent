"""
医学RAG系统 - 综合测试脚本
测试内容：
1. 向量数据库连接测试
2. 检索功能测试
3. API接口测试
4. 端到端RAG问答测试
5. 检索质量评估
"""

import os
import sys
import time
import json
import requests
from typing import List, Dict
import traceback

# --- 兼容性补丁：解决 Pydantic 2.x 路径问题 ---
try:
    from pydantic import v1 as pydantic_v1
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
    print("✅ Pydantic 兼容性补丁已加载")
except Exception as e:
    print(f"⚠️ 补丁加载失败: {e}")

from zhipuai import ZhipuAI
from model1 import ZhipuAIEmbeddings, create_deepseek_client, generate_deepseek_answer
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever

# ==================== 测试配置 ====================
API_URL = "http://localhost:8103"
MILVUS_QA_URI = "./milvus_agent.db"
MILVUS_PDF_URI = "./pdf_agent.db"
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')
if not ZHIPU_API_KEY:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")

# 测试用例集合
TEST_QUERIES = [
    "什么是糖尿病？",
    "糖尿病患者的饮食禁忌有哪些？",
    "牛皮癣和湿疹的区别是什么？",
    "血糖正常值是多少？",
    "如何预防医学微生物感染？"
]


class RAGSystemTester:
    """RAG系统测试器"""

    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "test_name": test_name,
            "status": status,
            "message": message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"{status} - {test_name}")
        if message:
            print(f"   └─ {message}")

    def test_1_database_connection(self):
        """测试1: 向量数据库连接"""
        print("\n" + "="*60)
        print("📊 测试1: 向量数据库连接")
        print("="*60)

        try:
            # 测试QA问答对数据库
            if not os.path.exists(MILVUS_QA_URI):
                self.log_test("QA数据库文件检查", False, f"文件不存在: {MILVUS_QA_URI}")
                return
            self.log_test("QA数据库文件检查", True, f"文件大小: {os.path.getsize(MILVUS_QA_URI)/1024/1024:.2f} MB")

            # 测试PDF文档数据库
            if not os.path.exists(MILVUS_PDF_URI):
                self.log_test("PDF数据库文件检查", False, f"文件不存在: {MILVUS_PDF_URI}")
                return
            self.log_test("PDF数据库文件检查", True, f"文件大小: {os.path.getsize(MILVUS_PDF_URI)/1024/1024:.2f} MB")

            # 测试数据库连接
            client_embedding = ZhipuAI(api_key=ZHIPU_API_KEY)
            embedding_model = ZhipuAIEmbeddings(client_embedding)

            milvus_qa = Milvus(
                embedding_function=embedding_model,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": MILVUS_QA_URI},
            )
            self.log_test("QA数据库连接", True)

            milvus_pdf = Milvus(
                embedding_function=embedding_model,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": MILVUS_PDF_URI},
            )
            self.log_test("PDF数据库连接", True)

            # 测试检索功能
            test_query = "糖尿病"
            results = milvus_qa.similarity_search(test_query, k=3)
            self.log_test("QA数据库检索测试", True, f"检索到 {len(results)} 条结果")

            pdf_results = milvus_pdf.similarity_search(test_query, k=3)
            self.log_test("PDF数据库检索测试", True, f"检索到 {len(pdf_results)} 条结果")

        except Exception as e:
            self.log_test("数据库连接", False, str(e))
            print(traceback.format_exc())

    def test_2_embedding_model(self):
        """测试2: 嵌入模型"""
        print("\n" + "="*60)
        print("🧠 测试2: 嵌入模型")
        print("="*60)

        try:
            client = ZhipuAI(api_key=ZHIPU_API_KEY)
            embeddings = ZhipuAIEmbeddings(client)

            test_text = "这是一个测试文本"
            start_time = time.time()
            vector = embeddings.embed_query(test_text)
            elapsed = time.time() - start_time

            self.log_test("智谱Embedding模型调用", True, f"向量维度: {len(vector)}, 耗时: {elapsed:.2f}s")

            # 批量嵌入测试
            texts = ["文本1", "文本2", "文本3"]
            start_time = time.time()
            vectors = embeddings.embed_documents(texts)
            elapsed = time.time() - start_time

            self.log_test("批量嵌入测试", True, f"嵌入 {len(texts)} 个文本, 耗时: {elapsed:.2f}s")

        except Exception as e:
            self.log_test("嵌入模型", False, str(e))
            print(traceback.format_exc())

    def test_3_llm_model(self):
        """测试3: 大语言模型"""
        print("\n" + "="*60)
        print("🤖 测试3: 大语言模型")
        print("="*60)

        try:
            llm = create_deepseek_client()
            test_prompt = "请用一句话解释什么是糖尿病。"

            start_time = time.time()
            response = generate_deepseek_answer(llm, test_prompt)
            elapsed = time.time() - start_time

            self.log_test("DeepSeek模型调用", True, f"耗时: {elapsed:.2f}s")
            print(f"   └─ 模型回复: {response.content[:100]}...")

        except Exception as e:
            self.log_test("大语言模型", False, str(e))
            print(traceback.format_exc())

    def test_4_api_server(self):
        """测试4: API服务"""
        print("\n" + "="*60)
        print("🌐 测试4: API服务")
        print("="*60)

        try:
            # 测试服务是否运行
            response = requests.get(f"{API_URL}/docs", timeout=5)
            if response.status_code == 200:
                self.log_test("API服务状态检查", True, "服务正常运行")
            else:
                self.log_test("API服务状态检查", False, f"状态码: {response.status_code}")
                return

            # 测试问答接口
            test_question = {"question": "什么是糖尿病？"}
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/",
                json=test_question,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                self.log_test("API问答接口", True, f"耗时: {elapsed:.2f}s")
                print(f"   └─ 回复: {result['response'][:100]}...")
            else:
                self.log_test("API问答接口", False, f"状态码: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.log_test("API服务状态检查", False, "服务未启动，请先运行 python agent.py")
        except Exception as e:
            self.log_test("API服务", False, str(e))
            print(traceback.format_exc())

    def test_5_retrieval_quality(self):
        """测试5: 检索质量评估"""
        print("\n" + "="*60)
        print("🎯 测试5: 检索质量评估")
        print("="*60)

        try:
            client = ZhipuAI(api_key=ZHIPU_API_KEY)
            embeddings = ZhipuAIEmbeddings(client)

            milvus_qa = Milvus(
                embedding_function=embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": MILVUS_QA_URI},
            )

            # 测试每个查询
            for query in TEST_QUERIES[:3]:  # 只测试前3个
                print(f"\n   🔍 查询: {query}")

                # 测试稠密向量检索
                results_dense = milvus_qa.similarity_search(
                    query,
                    k=3,
                    # ranker_type="rrf"  # 如果需要测试RRF
                )

                # 测试混合检索（稠密+稀疏）
                results_hybrid = milvus_qa.similarity_search(
                    query,
                    k=3,
                    ranker_type="rrf",
                    ranker_params={"k": 100}
                )

                print(f"   └─ 稠密检索召回: {len(results_dense)} 条")
                print(f"   └─ 混合检索召回: {len(results_hybrid)} 条")

                if results_hybrid:
                    print(f"   └─ 最相关文档片段: {results_hybrid[0].page_content[:80]}...")

            self.log_test("检索质量评估", True, "所有查询测试完成")

        except Exception as e:
            self.log_test("检索质量评估", False, str(e))
            print(traceback.format_exc())

    def test_6_end_to_end_rag(self):
        """测试6: 端到端RAG问答"""
        print("\n" + "="*60)
        print("🚀 测试6: 端到端RAG问答")
        print("="*60)

        try:
            client = ZhipuAI(api_key=ZHIPU_API_KEY)
            embeddings = ZhipuAIEmbeddings(client)
            llm = create_deepseek_client()

            # 初始化两路检索器
            milvus_qa = Milvus(
                embedding_function=embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": MILVUS_QA_URI},
            )

            docstore = InMemoryStore()
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

            milvus_pdf = Milvus(
                embedding_function=embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                connection_args={"uri": MILVUS_PDF_URI},
                consistency_level="Bounded",
                drop_old=False,
            )

            parent_retriever = ParentDocumentRetriever(
                vectorstore=milvus_pdf,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )

            # 测试完整RAG流程
            test_query = "糖尿病患者的饮食建议是什么？"
            print(f"\n   ❓ 问题: {test_query}")

            # 路径1: QA召回
            recall_rerank_milvus = milvus_qa.similarity_search(
                test_query,
                k=10,
                ranker_type="rrf",
                ranker_params={"k": 100}
            )
            print(f"   └─ QA召回: {len(recall_rerank_milvus)} 条")

            # 路径2: PDF召回
            retrieved_docs = parent_retriever.invoke(test_query)
            if retrieved_docs and len(retrieved_docs) >= 1:
                res = retrieved_docs[0].page_content
                print(f"   └─ PDF召回: {len(retrieved_docs)} 条")

            # 组合上下文
            context = "\n\n".join([doc.page_content for doc in recall_rerank_milvus])
            if retrieved_docs and len(retrieved_docs) >= 1:
                context = context + "\n" + res

            # 生成回复
            SYSTEM_PROMPT = "System:你是一个非常得力的医学助手,你可以通过从数据库中检索出的信息找到问题的答案。"
            USER_PROMPT = f"""User:利用介于<context>和</context>之间的从数据库中检索出的信息来回答问题,具体的问题介于<question>和</question>之间,如果提供的信息为空,则按照你的经验知识来给出尽可能严谨准确的回答,不知道的时候坦诚的承认不了解,不要编造不真实的信息。
<context>
{context[:2000]}
</context>
<question>
{test_query}
</question>
"""

            start_time = time.time()
            response = generate_deepseek_answer(llm, SYSTEM_PROMPT + USER_PROMPT)
            elapsed = time.time() - start_time

            print(f"\n   ✨ RAG回复 (耗时: {elapsed:.2f}s):")
            print(f"   └─ {response.content[:200]}...")

            self.log_test("端到端RAG流程", True)

        except Exception as e:
            self.log_test("端到端RAG流程", False, str(e))
            print(traceback.format_exc())

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "🔬 "*20)
        print("🎓 医学RAG系统 - 综合测试")
        print("🔬 "*20)

        self.test_1_database_connection()
        self.test_2_embedding_model()
        self.test_3_llm_model()
        self.test_4_api_server()
        self.test_5_retrieval_quality()
        self.test_6_end_to_end_rag()

        # 打印测试报告
        print("\n" + "="*60)
        print("📋 测试报告")
        print("="*60)
        total = self.passed + self.failed
        print(f"总测试数: {total}")
        print(f"✅ 通过: {self.passed}")
        print(f"❌ 失败: {self.failed}")
        print(f"通过率: {self.passed/total*100:.1f}%")

        # 保存测试报告
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 测试报告已保存到: test_report.json")


def main():
    """主函数"""
    tester = RAGSystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
