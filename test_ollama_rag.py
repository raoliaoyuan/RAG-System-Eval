#!/usr/bin/env python3
"""
测试Ollama RAG系统功能
"""

import requests
import json
import time

# 测试配置
BASE_URL = "http://localhost:8000"

def test_stats():
    """测试系统状态"""
    print("🔍 测试系统状态...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ 系统状态正常")
            print(f"   文档数量: {stats['document_count']}")
            print(f"   索引状态: {stats['is_initialized']}")
            print(f"   模型名称: {stats['model_name']}")
            return True
        else:
            print(f"❌ 获取系统状态失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接系统失败: {e}")
        return False

def test_upload():
    """测试文档上传"""
    print("\n📁 测试文档上传...")
    try:
        # 创建测试文档
        test_content = """
        人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
        
        机器学习是AI的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。
        
        深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。
        
        自然语言处理（NLP）是AI的一个重要领域，专注于使计算机能够理解、解释和生成人类语言。
        """
        
        with open("test_document.txt", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # 上传文档
        with open("test_document.txt", "rb") as f:
            files = {"files": ("test_document.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文档上传成功: {result['message']}")
            return True
        else:
            print(f"❌ 文档上传失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 文档上传异常: {e}")
        return False

def test_build_index():
    """测试索引构建"""
    print("\n🔧 测试索引构建...")
    try:
        response = requests.post(f"{BASE_URL}/build-index")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 索引构建成功: {result['message']}")
            return True
        else:
            print(f"❌ 索引构建失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 索引构建异常: {e}")
        return False

def test_qa():
    """测试问答功能"""
    print("\n❓ 测试问答功能...")
    test_questions = [
        "什么是人工智能？",
        "机器学习和深度学习有什么区别？",
        "自然语言处理的主要应用是什么？"
    ]
    
    for question in test_questions:
        print(f"\n   问题: {question}")
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"query": question},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"   ❌ 错误: {result['error']}")
                else:
                    print(f"   ✅ 答案: {result['answer'][:100]}...")
                    print(f"   📊 置信度: {result['confidence']:.2f}")
                    print(f"   📚 来源数量: {len(result['sources'])}")
            else:
                print(f"   ❌ 请求失败: {response.status_code}")
        except Exception as e:
            print(f"   ❌ 问答异常: {e}")
        
        time.sleep(1)  # 避免请求过快

def main():
    """主测试函数"""
    print("🚀 开始测试Ollama RAG系统...")
    print("=" * 50)
    
    # 测试系统状态
    if not test_stats():
        print("❌ 系统状态测试失败，请检查服务是否正常运行")
        return
    
    # 测试文档上传
    if not test_upload():
        print("❌ 文档上传测试失败")
        return
    
    # 测试索引构建
    if not test_build_index():
        print("❌ 索引构建测试失败")
        return
    
    # 测试问答功能
    test_qa()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")

if __name__ == "__main__":
    main() 