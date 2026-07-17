from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import uuid
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os

embedding_path = "models/bce-embedding"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_path,
    model_kwargs={'device': 'cpu'},  # 你的驱动旧，强制用 CPU
    encode_kwargs={'normalize_embeddings': True},
    # 这一行是关键：告诉它这是一个原始 Transformer 模型，不要去找 Pooling 文件夹
    cache_folder=None 
)

# 请通过环境变量设置API Key: export DEEPSEEK_API_KEY=your_key
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
if not deepseek_api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

# 初始化Deepseek模型
llm = ChatOpenAI(
    model = 'deepseek-chat',
    openai_api_key = deepseek_api_key,
    base_url = 'https://api.deepseek.com/v1',
    temperature = 0.7,
    max_tokens = 2048
)

docs = [
    Document(page_content="爸爸前⼏天去医院刚检查出有糖尿病,今年60岁⾎糖10请问专家吃什么能好转?求各位告诉我⼀下谢谢? \
医⽣回复:⾸先，建议您的⽗亲遵循医⽣的建议，按时服药和定期检查⾎糖。其次，饮⻝对于糖尿病患者⾮常重要，建议他遵循以下饮⻝原则：\n \
1. 控制碳⽔化合物的摄⼊量，尤其是⾼糖、⾼淀粉的⻝物，例如⽩⽶饭、⾯条、⾯包、糖果、饮料等。\n \
2. 增加蔬菜和⽔果的摄⼊量，尤其是低糖、低淀粉的蔬菜，例如菠菜、芦笋、西兰花、⾖腐等，以及⽔果中的柚⼦、苹果、草莓、蓝莓等。\n \
3. 适量增加优质蛋⽩质的摄⼊量，例如鸡蛋、⿂、⾖类、瘦⾁等。\n \
4. 避免⾷⽤⾼脂、⾼盐的⻝物，例如猪⾁、⽜⾁、奶油、蛋糕等。\n \
5. 控制饮⻝量和饮⻝频率，避免⼤量进⻝和过度饥饿。\n \
6. 坚持规律的饮⻝习惯，每⽇三餐定时定量，避免随意进⻝和过度饥饿。\n \
此外，适量运动也有助于控制⾎糖，建议您的⽗亲每天进⾏适当的运动，例如散步、慢跑、瑜伽等。最后，建议您的⽗亲定期复查⾎糖，及时调整治疗⽅案。",
             metadata={"doc_id": str(uuid.uuid4())}),
    Document(page_content="不⼩⼼得了⽜⽪癣了，但是不知道有什么特点。感觉和湿疹差不多。都很痒，反反复复的。⼀直好不了。\
⽜⽪癣和湿疹都是常⻅的⽪肤病，但两者的病因和治疗⽅法不同。⽜⽪癣是⼀种⾃身免疫性疾病，主要表现为⽪肤出现红斑、鳞屑、瘙痒、疼痛等症状。\
⽽湿疹则是⼀种⾮传染性⽪肤炎症，表现为⽪肤红肿、瘙痒、渗出、结痂等症状。\n \
如果您怀疑⾃⼰得了⽜⽪癣，建议尽快就医。医⽣会根据病情给出相应的治疗⽅案，包括外⽤药、⼝服药、光疗等。\
同时，保持⽪肤清洁、避免刺激、保持⼼情舒畅也有助于缓解症状。",
             metadata={"doc_id": str(uuid.uuid4())}),
]

doc_ids = [doc.metadata["doc_id"] for doc in docs]
print('doc_ids:', doc_ids)

question_gen_prompt_str = (
    "你是⼀位AI医学专家。请根据以下⽂档内容,⽣成3个⽤户可能会提出的,⾼度相关的问题。\n"
    "只返回问题列表，每个问题占⼀⾏，不要有其他前缀或编号。\n"
    "⽂档内容:\n"
    "----------\n"
    "{content}\n"
    "----------\n"
)

question_gen_prompt = ChatPromptTemplate.from_template(question_gen_prompt_str)
question_generator_chain = question_gen_prompt | llm | StrOutputParser()

sub_docs = []
for i, doc in enumerate(docs):
    doc_id = doc_ids[i]
    generated_questions = question_generator_chain.invoke({"content":doc.page_content}).split("\n")
    generated_questions = [q.strip() for q in generated_questions if q.strip()]

    for q in generated_questions:
        sub_docs.append(Document(page_content=q, metadata={"doc_id":doc_id}))

print("创建Chroma向量数据库, 并添加⽂档...")
vectorstore = Chroma.from_documents(documents=sub_docs, embedding=embeddings)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3 # 检索3个结果

# 初始化向量检索器 (Dense Retriever)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 初始化EnsembleRetriever, 将BM25和向量检索器结合
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever, vector_retriever],
    weights = [0.4, 0.6]
)

reranker_path = "models/bce-reranker"
# 引入reranker模型
model = HuggingFaceCrossEncoder(model_name=reranker_path, model_kwargs={'device':'cpu'})

compressor = CrossEncoderReranker(model=model, top_n=3)

print('初始化ContextualCompressionRetriever...')
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor,
    base_retriever = ensemble_retriever
)

compressed_docs = compression_retriever.invoke("糖人的饮食禁忌？")
print(compressed_docs)