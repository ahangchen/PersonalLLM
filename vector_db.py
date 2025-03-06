# 使用DeepSeek生成向量
from Milvus import Milvus
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self):
        self.vector_db = Milvus(collection_name="papers")
        # 加载轻量模型（英文）
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB     
        self.bm25_retriever = None
        
def chunk2vector(self, chunks):
    vectors = []
    for chunk in chunks:
        emb = self.model.encode(chunks, convert_to_numpy=True)
        vectors.append({
            "text": chunk,
            "vector": emb,
            "metadata": {"paper_id": "1234"}
        })
    return vectors

def vector2storage(self, vectors):
    # 构建多级索引
    self.vector_db.insert(vectors)

def chunk2index(self, chunks):
    self.bm25_retriever = BM25Retriever.from_texts(chunks)

def insert(self, chunks):
    vectors = self.chunk2vector(chunks)
    self.vector2storage(vectors)
    self.chunk2index(chunks)

# 混合检索策略
def hybrid_search(self, query):
    vector_results = self.vector_db.similarity_search(query, k=10)
    bm25_results = self.bm25_retriever.get_relevant_documents(query)[:5]
    return rerank(query, vector_results + bm25_results)