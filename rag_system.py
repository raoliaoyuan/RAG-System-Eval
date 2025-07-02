import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import openai
import requests
from config import Config

class OllamaEmbeddings:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            vectors.append(self.embed_query(text))
        return vectors

    def embed_query(self, text):
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embedding API 调用失败: {e}")

class RAGSystem:
    def __init__(self, embedding_config=None, **kwargs):
        if embedding_config is None:
            embedding_config = Config.get_ollama_embedding_config()
        self.model_name = embedding_config["model"]
        self.use_offline = kwargs.get('use_offline', False)  # 默认使用在线Ollama
        
        # 尝试初始化嵌入模型
        try:
            if self.use_offline:
                # 使用本地模型或简单的TF-IDF作为备选
                print("使用离线模式，初始化简单嵌入模型...")
                self.embeddings = self._create_simple_embeddings()
            else:
                print(f"尝试连接Ollama embedding模型: {self.model_name}")
                self.embeddings = OllamaEmbeddings(
                    base_url=embedding_config["base_url"],
                    model=embedding_config["model"]
                )
        except Exception as e:
            print(f"Ollama embedding模型连接失败: {e}")
            print("使用简单嵌入模型作为备选...")
            self.embeddings = self._create_simple_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.documents = []
        self.embeddings_matrix = None
        self.index = None
        self.is_initialized = False
        
    def _create_simple_embeddings(self):
        """创建简单的嵌入模型作为备选"""
        class SimpleEmbeddings:
            def __init__(self):
                self.dimension = 384  # 固定维度
                
            def embed_documents(self, texts):
                """简单的文档嵌入实现"""
                embeddings = []
                for text in texts:
                    # 使用简单的哈希方法创建向量
                    import hashlib
                    hash_obj = hashlib.md5(text.encode())
                    hash_bytes = hash_obj.digest()
                    
                    # 将哈希转换为向量
                    vector = []
                    for i in range(0, len(hash_bytes), 4):
                        if len(vector) >= self.dimension:
                            break
                        chunk = hash_bytes[i:i+4]
                        value = int.from_bytes(chunk, byteorder='big')
                        vector.append((value % 1000) / 1000.0)  # 归一化到0-1
                    
                    # 填充到固定维度
                    while len(vector) < self.dimension:
                        vector.append(0.0)
                    vector = vector[:self.dimension]
                    
                    embeddings.append(vector)
                return embeddings
                
            def embed_query(self, text):
                """查询嵌入"""
                return self.embed_documents([text])[0]
        
        return SimpleEmbeddings()
    
    def load_document(self, file_path: str) -> List[str]:
        """
        加载文档并分割成块
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档块列表
        """
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return [chunk.page_content for chunk in chunks]
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            return [chunk.page_content for chunk in chunks]
        elif file_extension == 'txt':
            # 直接读取文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = self.text_splitter.split_text(content)
            return chunks
        else:
            raise ValueError(f"不支持的文件格式: {file_extension}")
    
    def add_documents(self, file_paths: List[str]):
        """
        添加多个文档到系统
        
        Args:
            file_paths: 文档路径列表
        """
        for file_path in file_paths:
            try:
                chunks = self.load_document(file_path)
                self.documents.extend(chunks)
                print(f"成功加载文档: {file_path}, 添加了 {len(chunks)} 个文本块")
            except Exception as e:
                print(f"加载文档失败 {file_path}: {str(e)}")
    
    def build_index(self):
        """
        构建向量索引
        """
        if not self.documents:
            raise ValueError("没有文档可以索引")
            
        print("开始构建向量索引...")
        
        # 计算文档嵌入
        embeddings_list = self.embeddings.embed_documents(self.documents)
        self.embeddings_matrix = np.array(embeddings_list).astype('float32')
        
        # 创建FAISS索引
        dimension = self.embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引，用于余弦相似度
        self.index.add(self.embeddings_matrix)
        
        self.is_initialized = True
        print(f"索引构建完成，包含 {len(self.documents)} 个文档块")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        if not self.is_initialized:
            raise ValueError("索引尚未构建，请先调用 build_index()")
            
        # 计算查询嵌入
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 搜索相似文档
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    'content': self.documents[idx],
                    'score': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def save_index(self, file_path: str):
        """
        保存索引到文件
        
        Args:
            file_path: 保存路径
        """
        if not self.is_initialized:
            raise ValueError("索引尚未构建")
            
        data = {
            'documents': self.documents,
            'embeddings_matrix': self.embeddings_matrix,
            'index': self.index,
            'model_name': self.model_name
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"索引已保存到: {file_path}")
    
    def load_index(self, file_path: str):
        """
        从文件加载索引
        
        Args:
            file_path: 索引文件路径
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        self.documents = data['documents']
        self.embeddings_matrix = data['embeddings_matrix']
        self.index = data['index']
        self.model_name = data['model_name']
        self.is_initialized = True
        
        print(f"索引已加载，包含 {len(self.documents)} 个文档块")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'document_count': len(self.documents),
            'is_initialized': self.is_initialized,
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else None,
            'use_offline': self.use_offline
        } 