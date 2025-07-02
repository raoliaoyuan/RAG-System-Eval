#!/usr/bin/env python3
"""
检查Ollama可用模型
"""

import requests
import json

def check_ollama_models():
    """检查Ollama可用的模型"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            print("🤖 Ollama可用模型:")
            for model in data.get('models', []):
                print(f"   - {model['name']} ({model['size']} bytes)")
            return data.get('models', [])
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 连接Ollama失败: {e}")
        return []

def test_model_generation(model_name):
    """测试模型生成能力"""
    print(f"\n🧪 测试模型: {model_name}")
    
    try:
        # 测试简单的生成请求
        payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 生成成功")
            print(f"   📝 响应: {result.get('response', 'N/A')[:100]}...")
            return True
        else:
            print(f"   ❌ 生成失败: {response.status_code}")
            print(f"   📄 错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ 测试异常: {e}")
        return False

def main():
    """主函数"""
    print("🔍 检查Ollama模型状态...")
    
    # 检查可用模型
    models = check_ollama_models()
    
    if not models:
        print("\n💡 建议:")
        print("   1. 确保Ollama服务正在运行")
        print("   2. 下载一个LLM模型，例如:")
        print("      ollama pull llama3")
        print("      ollama pull qwen2.5:7b")
        print("      ollama pull gemma2:2b")
        return
    
    # 测试第一个模型（如果有的话）
    if models:
        first_model = models[0]['name']
        test_model_generation(first_model)
    
    print("\n💡 配置建议:")
    print("   1. 在RAG系统中使用可用的模型名称")
    print("   2. 确保模型支持文本生成功能")
    print("   3. 如果只有embedding模型，需要下载LLM模型")

if __name__ == "__main__":
    main() 