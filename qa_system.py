import requests
from config import Config
from rag_system import RAGSystem
from typing import List, Dict, Any
import time

class OllamaLLM:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama LLM API 调用失败: {e}")

class QASystem:
    def __init__(self, rag_system: RAGSystem, llm_config=None):
        self.rag_system = rag_system
        self.llm_config = llm_config or {}

    def _call_ollama_llm(self, prompt: str) -> str:
        """调用Ollama LLM API"""
        try:
            # 检查是否有可用的LLM模型
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                llm_models = [m for m in models if 'embed' not in m['name'].lower()]
                
                if not llm_models:
                    return "抱歉，当前Ollama服务中没有可用的LLM模型。请先下载一个LLM模型，例如：\n" + \
                           "1. ollama pull llama3\n" + \
                           "2. ollama pull qwen2.5:7b\n" + \
                           "3. ollama pull gemma2:2b\n" + \
                           "或者配置使用OpenAI API等其他LLM服务。"
            
            # 尝试使用配置的模型
            model_name = self.llm_config.get('model', 'llama3')
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.llm_config.get('base_url', 'http://localhost:11434')}/api/generate",
                json=payload,
                timeout=120  # 增加超时时间到120秒
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                error_msg = f"Ollama LLM API 调用失败: {response.status_code} {response.reason}"
                try:
                    error_detail = response.json().get('error', '')
                    if error_detail:
                        error_msg += f" - {error_detail}"
                except:
                    pass
                return f"生成答案时出错: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "生成答案时出错: LLM API 请求超时（120秒），模型可能正在加载中，请稍后再试"
        except requests.exceptions.ConnectionError:
            return "生成答案时出错: 无法连接到Ollama服务，请确保Ollama正在运行"
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def _call_openai_llm(self, prompt: str) -> str:
        """调用OpenAI兼容的LLM API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config.get('api_key', '')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.llm_config.get('model', 'gpt-3.5-turbo'),
                "messages": [
                    {"role": "system", "content": "你是一个有用的AI助手，请基于提供的上下文信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.llm_config.get('base_url', 'https://api.openai.com/v1')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_msg = f"OpenAI API 调用失败: {response.status_code} {response.reason}"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f" - {error_detail}"
                except:
                    pass
                return f"生成答案时出错: {error_msg}"
                
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def _call_custom_llm(self, prompt: str) -> str:
        """调用自定义LLM API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llm_config.get('api_key', '')}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.llm_config.get('model', ''),
                "messages": [
                    {"role": "system", "content": "你是一个有用的AI助手，请基于提供的上下文信息回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.llm_config.get('api_url', ''),
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_msg = f"自定义API 调用失败: {response.status_code} {response.reason}"
                try:
                    error_detail = response.json().get('error', {}).get('message', '')
                    if error_detail:
                        error_msg += f" - {error_detail}"
                except:
                    pass
                return f"生成答案时出错: {error_msg}"
                
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def _generate_answer(self, query: str, context: str) -> str:
        """生成答案"""
        provider = self.llm_config.get('provider', 'ollama')
        
        # 构建提示词
        prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的信息中找到答案。

上下文信息:
{context}

问题: {query}

请提供准确、详细的回答:"""

        # 根据提供商调用相应的API
        if provider == 'ollama':
            return self._call_ollama_llm(prompt)
        elif provider == 'openai':
            return self._call_openai_llm(prompt)
        elif provider == 'custom':
            return self._call_custom_llm(prompt)
        else:
            return "生成答案时出错: 不支持的LLM提供商"
    
    def get_answer_with_sources(self, query: str) -> Dict[str, Any]:
        """获取答案和来源"""
        start_time = time.time()
        
        try:
            # 检查RAG系统是否已初始化
            if not self.rag_system.is_initialized:
                return {
                    "query": query,
                    "answer": "RAG系统尚未初始化，请先添加文档并构建索引",
                    "sources": [],
                    "confidence": 0.0,
                    "response_time": 0
                }
            
            # 检索相关文档
            results = self.rag_system.search(query, top_k=5)
            
            if not results:
                return {
                    "query": query,
                    "answer": "抱歉，没有找到与您问题相关的文档信息。",
                    "sources": [],
                    "confidence": 0.0,
                    "response_time": 0
                }
            
            # 构建上下文
            context = "\n\n".join([f"文档片段 {i+1}: {result['content']}" for i, result in enumerate(results)])
            
            # 生成答案
            answer = self._generate_answer(query, context)
            
            # 计算响应时间
            response_time = (time.time() - start_time) * 1000
            
            # 计算置信度（基于检索结果的相似度分数）
            confidence = sum(result['score'] for result in results) / len(results) if results else 0.0
            
            # 格式化来源信息
            sources = []
            for i, result in enumerate(results):
                sources.append({
                    "rank": i + 1,
                    "content": result['content'],
                    "score": result['score'],
                    "metadata": result.get('metadata', {})
                })
            
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "response_time": response_time
            }
            
        except Exception as e:
            return {
                "query": query,
                "answer": f"处理问题时出错: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "response_time": (time.time() - start_time) * 1000
            } 