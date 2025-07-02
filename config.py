import os
from typing import Dict, Any

class Config:
    # Ollama本地部署相关配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:1.5b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-small-zh")

    @staticmethod
    def get_ollama_llm_config():
        return {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "deepseek-r1:1.5b"  # 使用实际可用的模型
        }
    
    @staticmethod
    def get_ollama_embedding_config():
        return {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "bge-small-zh"
        }
    
    @staticmethod
    def get_openai_llm_config():
        return {
            "provider": "openai",
            "api_key": "your-openai-api-key",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1"
        }
    
    @staticmethod
    def get_openai_embedding_config():
        return {
            "provider": "openai",
            "api_key": "your-openai-api-key",
            "model": "text-embedding-ada-002",
            "base_url": "https://api.openai.com/v1"
        } 