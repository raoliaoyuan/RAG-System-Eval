#!/usr/bin/env python3
"""
RAG系统完整功能演示
包括模型配置、文档处理、问答等功能
"""

import requests
import json
import time
import os

# 测试配置
BASE_URL = "http://localhost:8000"

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*20} {title} {'='*20}")

def print_step(step, description):
    """打印步骤信息"""
    print(f"\n{step}. {description}")

def wait_for_service():
    """等待服务启动"""
    print_step("0", "等待服务启动...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/stats", timeout=5)
            if response.status_code == 200:
                print("   ✅ 服务已启动")
                return True
        except:
            pass
        print(f"   ⏳ 等待服务启动... ({i+1}/{max_retries})")
        time.sleep(2)
    print("   ❌ 服务启动超时")
    return False

def test_model_configuration():
    """测试模型配置功能"""
    print_section("模型配置测试")
    
    # 测试Ollama配置
    print_step("1", "测试Ollama本地模型配置")
    ollama_config = {
        "llm_config": {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3"
        },
        "embedding_config": {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "bge-small-zh"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/apply-config",
            json=ollama_config,
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        if result['success']:
            print("   ✅ Ollama配置应用成功")
        else:
            print(f"   ❌ Ollama配置失败: {result['error']}")
    except Exception as e:
        print(f"   ❌ Ollama配置异常: {e}")
    
    # 测试OpenAI配置（模拟）
    print_step("2", "测试OpenAI API配置（模拟）")
    openai_config = {
        "llm_config": {
            "provider": "openai",
            "api_key": "sk-demo-key",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.openai.com/v1"
        },
        "embedding_config": {
            "provider": "openai",
            "api_key": "sk-demo-key",
            "model": "text-embedding-ada-002",
            "base_url": "https://api.openai.com/v1"
        }
    }
    
    print("   📝 注意: 这是模拟配置，需要真实API Key才能实际使用")
    print("   💡 可以在Web界面中配置真实的OpenAI API")
    
    # 测试自定义API配置（模拟）
    print_step("3", "测试自定义API配置（模拟）")
    custom_config = {
        "llm_config": {
            "provider": "custom",
            "api_url": "https://api.example.com/v1/chat/completions",
            "api_key": "custom-key",
            "model": "custom-model"
        },
        "embedding_config": {
            "provider": "custom",
            "api_url": "https://api.example.com/v1/embeddings",
            "api_key": "custom-key",
            "model": "custom-embedding"
        }
    }
    
    print("   📝 注意: 这是模拟配置，需要真实API端点才能实际使用")
    print("   💡 可以在Web界面中配置自定义API端点")

def test_document_processing():
    """测试文档处理功能"""
    print_section("文档处理测试")
    
    # 检查测试文档
    test_doc = "enhanced_test_document.txt"
    if not os.path.exists(test_doc):
        print(f"   ❌ 测试文档 {test_doc} 不存在")
        return False
    
    print_step("1", f"上传测试文档: {test_doc}")
    
    try:
        with open(test_doc, 'rb') as f:
            files = {'files': (test_doc, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            result = response.json()
            print(f"   ✅ {result['message']}")
    except Exception as e:
        print(f"   ❌ 文档上传失败: {e}")
        return False
    
    print_step("2", "构建向量索引")
    try:
        response = requests.post(f"{BASE_URL}/build-index")
        result = response.json()
        print(f"   ✅ {result['message']}")
    except Exception as e:
        print(f"   ❌ 索引构建失败: {e}")
        return False
    
    return True

def test_qa_functionality():
    """测试问答功能"""
    print_section("问答功能测试")
    
    # 测试问题列表
    test_questions = [
        "什么是人工智能？",
        "机器学习有哪些主要类型？",
        "深度学习与传统机器学习有什么区别？",
        "什么是神经网络？",
        "强化学习的基本原理是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print_step(f"{i}", f"提问: {question}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"query": question},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    response_time = (end_time - start_time) * 1000
                    print(f"   ✅ 回答成功")
                    print(f"      - 响应时间: {response_time:.0f}ms")
                    print(f"      - 答案长度: {len(result['answer'])} 字符")
                    print(f"      - 来源数量: {len(result['sources'])}")
                    print(f"      - 置信度: {result['confidence']:.2f}")
                    print(f"      - 答案预览: {result['answer'][:100]}...")
                else:
                    print(f"   ❌ 回答错误: {result['error']}")
            else:
                print(f"   ❌ 请求失败: {response.status_code}")
        except Exception as e:
            print(f"   ❌ 问答异常: {e}")
        
        print()  # 空行分隔

def test_system_metrics():
    """测试系统指标"""
    print_section("系统指标测试")
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("   📊 系统状态:")
            print(f"      - 文档数量: {stats['document_count']}")
            print(f"      - 索引状态: {'✅ 已初始化' if stats['is_initialized'] else '❌ 未初始化'}")
            print(f"      - 嵌入模型: {stats['model_name']}")
            print(f"      - 向量维度: {stats.get('embedding_dimension', 'N/A')}")
        else:
            print(f"   ❌ 获取系统状态失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 系统状态异常: {e}")

def show_ui_features():
    """展示UI功能"""
    print_section("用户界面功能")
    
    print("🎨 新的用户界面特性:")
    print("   1. 📱 响应式布局 - 左侧边栏 + 右侧主内容区")
    print("   2. ⚙️  模型配置面板 - 支持多种提供商切换")
    print("      - 🤖 本地 Ollama 模型")
    print("      - 🔌 OpenAI 兼容 API")
    print("      - 🔗 自定义 API 端点")
    print("   3. 🧪 连接测试功能 - 实时验证模型连接")
    print("   4. 📊 详细系统指标 - 响应时间、置信度等")
    print("   5. 🎯 结构化输出 - 问题、答案、来源分离显示")
    print("   6. 🔄 实时状态更新 - 自动刷新系统状态")
    
    print("\n💡 使用提示:")
    print("   1. 访问 http://localhost:8000 打开Web界面")
    print("   2. 在左侧配置面板中选择模型提供商")
    print("   3. 输入相应的API配置信息")
    print("   4. 点击测试连接验证配置")
    print("   5. 应用配置后上传文档并构建索引")
    print("   6. 在右侧进行智能问答")

def main():
    """主演示函数"""
    print("🚀 RAG系统完整功能演示")
    print("=" * 60)
    
    # 等待服务启动
    if not wait_for_service():
        return
    
    # 测试模型配置
    test_model_configuration()
    
    # 测试文档处理
    if test_document_processing():
        # 测试问答功能
        test_qa_functionality()
    
    # 测试系统指标
    test_system_metrics()
    
    # 展示UI功能
    show_ui_features()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("🌐 访问 http://localhost:8000 体验完整功能")

if __name__ == "__main__":
    main() 