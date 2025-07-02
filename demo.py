#!/usr/bin/env python3
"""
RAG系统演示脚本
"""

import os
import sys
from rag_system import RAGSystem
from qa_system import QASystem

def create_sample_documents():
    """创建示例文档"""
    sample_docs = {
        "ai_introduction.txt": """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、
语言识别、图像识别、自然语言处理和专家系统等。

人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能
带来的科技产品，将会是人类智慧的"容器"。人工智能可以对人的意识、思维的信息过程的模拟。
人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。

人工智能是一门极富挑战性的科学，从事人工智能工作的人员必须懂得计算机知识，心理学和哲学。
人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习，计算机视觉等等，总的说来，
人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。
        """,
        
        "machine_learning.txt": """
机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。
机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。

机器学习主要分为三种类型：
1. 监督学习：使用标记的训练数据来学习输入和输出之间的映射关系
2. 无监督学习：从未标记的数据中发现隐藏的模式和结构
3. 强化学习：通过与环境交互来学习最优的行为策略

常见的机器学习算法包括：
- 线性回归
- 决策树
- 随机森林
- 支持向量机
- 神经网络
- 深度学习

机器学习在各个领域都有广泛应用，包括：
- 推荐系统
- 图像识别
- 自然语言处理
- 医疗诊断
- 金融预测
        """,
        
        "deep_learning.txt": """
深度学习是机器学习的一个分支，它基于人工神经网络，特别是深度神经网络。
深度学习模型可以自动学习数据的层次化表示，这使得它在处理复杂模式识别任务时非常有效。

深度学习的主要特点：
1. 多层神经网络结构
2. 自动特征提取
3. 端到端学习
4. 需要大量数据训练

常见的深度学习架构：
- 卷积神经网络（CNN）：主要用于图像处理
- 循环神经网络（RNN）：适用于序列数据
- 长短期记忆网络（LSTM）：改进的RNN，解决梯度消失问题
- 变换器（Transformer）：在自然语言处理中表现出色

深度学习的应用领域：
- 计算机视觉
- 语音识别
- 自然语言处理
- 游戏AI
- 自动驾驶
        """
    }
    
    # 创建示例文档目录
    if not os.path.exists("sample_docs"):
        os.makedirs("sample_docs")
    
    # 写入示例文档
    for filename, content in sample_docs.items():
        filepath = os.path.join("sample_docs", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"创建示例文档: {filename}")
    
    return list(sample_docs.keys())

def main():
    """主演示函数"""
    print("🔍 RAG系统演示")
    print("=" * 50)
    
    # 创建示例文档
    print("\n1. 创建示例文档...")
    sample_files = create_sample_documents()
    sample_paths = [os.path.join("sample_docs", f) for f in sample_files]
    
    # 初始化RAG系统
    print("\n2. 初始化RAG系统...")
    rag_system = RAGSystem()
    qa_system = QASystem(rag_system)
    
    # 添加文档
    print("\n3. 添加文档到系统...")
    rag_system.add_documents(sample_paths)
    
    # 构建索引
    print("\n4. 构建向量索引...")
    rag_system.build_index()
    
    # 显示系统状态
    print("\n5. 系统状态:")
    stats = rag_system.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 演示问答
    print("\n6. 演示问答功能:")
    demo_questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "深度学习的主要特点是什么？",
        "神经网络在哪些领域有应用？"
    ]
    
    for question in demo_questions:
        print(f"\n问题: {question}")
        print("-" * 30)
        
        result = qa_system.get_answer_with_sources(question)
        
        if "error" in result:
            print(f"错误: {result['error']}")
        else:
            print(f"答案: {result['answer']}")
            print(f"置信度: {result['confidence']:.3f}")
            print("来源文档:")
            for i, source in enumerate(result['sources'][:2], 1):
                print(f"  {i}. 相似度: {source['score']:.3f}")
                print(f"     {source['content'][:100]}...")
    
    # 交互式问答
    print("\n" + "=" * 50)
    print("🎯 交互式问答模式")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("正在思考...")
            result = qa_system.get_answer_with_sources(question)
            
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                print(f"\n✅ 答案: {result['answer']}")
                print(f"📊 置信度: {result['confidence']:.3f}")
                
                if result['sources']:
                    print("\n📚 相关文档:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. 相似度: {source['score']:.3f}")
                        print(f"     {source['content'][:150]}...")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 