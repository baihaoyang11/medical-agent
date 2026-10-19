import os
import sys
import uuid  # 讲义第2页要求：用于给文档加ID，防止幻觉回溯失败 [cite: 20, 27]
from openai import OpenAI

# --- 兼容性补丁：解决 Pydantic 2.x 路径问题 ---
try:
    from pydantic import v1 as pydantic_v1
    # 创建正确的模块映射
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
    print("✅ Pydantic 兼容性补丁已加载 (V2 -> V1 映射成功)")
except Exception as e:
    print(f"⚠️ 补丁加载失败: {e}")

from zhipuai import ZhipuAI
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document  # 封装医疗文档的标准格式 [cite: 44, 316]

from langchain_milvus import Milvus, BM25BuiltInFunction  # Milvus 核心 [cite: 544, 670]
from langchain_community.retrievers import BM25Retriever  # 关键词检索器 

# 请通过环境变量设置API Key: export DEEPSEEK_API_KEY=your_key
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
if not deepseek_api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

# 请通过环境变量设置智谱AI API Key: export ZHIPU_API_KEY=your_key
zhipu_api_key = os.getenv('ZHIPU_API_KEY')
if not zhipu_api_key:
    raise ValueError("请设置环境变量 ZHIPU_API_KEY")
client = ZhipuAI(api_key=zhipu_api_key)

class ZhipuAIEmbeddings(Embeddings):
    def __init__(self,client):
        self.client = client

    def embed_documents(self,texts):
        embeddings = []

        for text in texts:
            response = self.client.embeddings.create(
                model = 'embedding-3',
                input = [text]
            )
            embeddings.append(response.data[0].embedding)

        return embeddings
    
    def embed_query(self,text):
        return self.embed_documents([text])[0]
    
# model2:大语言模型部署(deepseek-chat)进行问题生成和答案生成
def create_deepseek_client():
    """医疗问答的温度必须低。

    temperature=0.7 适合创作，但医学问答要的是可复现、少发挥，
    默认 0.2；需要更高稳定性可通过环境变量调到 0。
    """
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm = ChatOpenAI(
        model = os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key = deepseek_api_key,
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        temperature = temperature,
        timeout = float(os.getenv("LLM_TIMEOUT", "60")),
        max_retries = 0,  # 重试由 core.retry 统一控制，避免两层重试叠加放大
    )
    print(f"Deepseek-chat模型创建成功 (temperature={temperature})")
    return llm

def generate_deepseek_answer(question,llm):
    answer = llm.invoke(question)
    return answer

if __name__ == '__main__':
    llm = create_deepseek_client()
    output = generate_deepseek_answer("什么是糖尿病？", llm)
    print("问题：什么是糖尿病？")
    print("答案：", output)