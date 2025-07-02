# 🔍 RAG演示系统

一个完整的检索增强生成（Retrieval-Augmented Generation, RAG）系统演示，支持文档上传、向量化检索和智能问答。

---

## ⚡ 主要特性

- 📁 **文档处理**: 支持PDF、DOCX、TXT文档上传和解析
- 🔍 **语义检索**: 基于向量相似度的智能文档检索
- 🤖 **智能问答**: 结合检索结果和LLM的问答系统
- 🌐 **Web界面**: 现代化Web界面，支持交互式问答
- 💻 **命令行工具**: 提供命令行演示
- 📊 **实时统计**: 显示系统状态和检索置信度

---

## 🚀 快速开始

### 环境要求
- Python 3.9 及以上
- 推荐使用虚拟环境

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Web界面

```bash
python app.py
```

浏览器访问: http://localhost:8000

### 3. 启动命令行演示

```bash
python demo.py
```

---

## 📦 依赖说明

主要依赖包（requirements.txt 已包含）：
- fastapi, uvicorn, python-multipart
- langchain, langchain-community, langchain-openai
- sentence-transformers, faiss-cpu
- pypdf, python-docx, docx2txt
- openai, streamlit, pandas, numpy

---

## 📖 使用指南

### Web界面
1. 上传文档（支持PDF、DOCX、TXT）
2. 构建索引
3. 输入问题，获取智能答案

### 命令行
见 demo.py 示例

---

## 🏗️ 目录结构

```
RAG演示系统
├── rag_system.py      # 核心RAG系统
├── qa_system.py       # 问答系统
├── app.py            # Web应用
├── demo.py           # 命令行演示
├── requirements.txt  # 依赖包
├── uploads/          # 上传文档目录
├── sample_docs/      # 示例文档
└── ...
```

---

## 🤝 如何贡献

欢迎任何形式的贡献！
1. Fork 本仓库并新建分支
2. 提交 Pull Request，描述你的更改
3. 如有问题请提 Issue

建议：
- 遵循PEP8风格
- 保持README和注释同步
- 不要上传大文件、数据集、个人密钥

---

## 📝 常见问题
- 启动报端口占用：请关闭占用8000端口的进程或更换端口
- LLM模型不可用：请检查Ollama或OpenAI配置
- 依赖安装失败：请确认Python版本和pip源

---

## 📄 许可证

MIT License

---

## 📬 联系方式

如有建议或合作意向，请通过Issue联系。

---

## 🔗 相关链接

- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

> 本项目已内置.gitignore，自动忽略临时文件、缓存、测试输出等，适合直接开源。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Web演示

```bash
python app.py
```

然后在浏览器中访问: http://localhost:8000

### 3. 运行命令行演示

```bash
python demo.py
```

## 📖 使用指南

### Web界面使用

1. **上传文档**: 在"文档上传"区域选择PDF或DOCX文件
2. **构建索引**: 点击"构建向量索引"按钮
3. **开始问答**: 在"智能问答"区域输入问题

### 命令行使用

```python
from rag_system import RAGSystem
from qa_system import QASystem

# 初始化系统
rag_system = RAGSystem()
qa_system = QASystem(rag_system)

# 添加文档
rag_system.add_documents(["document1.pdf", "document2.docx"])

# 构建索引
rag_system.build_index()

# 提问
result = qa_system.get_answer_with_sources("你的问题")
print(result['answer'])
```

## 🏗️ 系统架构

```
RAG演示系统
├── rag_system.py      # 核心RAG系统
├── qa_system.py       # 问答系统
├── app.py            # Web应用
├── demo.py           # 命令行演示
└── requirements.txt  # 依赖包
```

### 核心组件

- **RAGSystem**: 文档处理、向量化和检索
- **QASystem**: 基于检索结果的问答生成
- **Web界面**: FastAPI + HTML/JavaScript
- **向量索引**: FAISS + Sentence Transformers

## 🔧 配置选项

### 环境变量

```bash
# OpenAI API密钥（可选，用于更好的问答质量）
export OPENAI_API_KEY="your-api-key"
```

### 模型配置

```python
# 使用不同的嵌入模型
rag_system = RAGSystem(model_name="all-mpnet-base-v2")

# 调整文本分割参数
rag_system.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 块大小
    chunk_overlap=100, # 重叠大小
)
```

## 📊 性能优化

- **批量处理**: 支持批量文档上传和索引构建
- **向量缓存**: 自动保存和加载向量索引
- **内存优化**: 使用高效的FAISS索引

## 🛠️ 开发指南

### 添加新的文档格式

```python
# 在rag_system.py中添加新的加载器
if file_extension == 'txt':
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = self.text_splitter.split_text(content)
```

### 自定义检索策略

```python
# 修改检索参数
results = rag_system.search(query, top_k=10)  # 检索更多文档
```

## 📝 示例文档

系统包含以下示例文档：
- `ai_introduction.txt`: 人工智能介绍
- `machine_learning.txt`: 机器学习基础
- `deep_learning.txt`: 深度学习概述

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🔗 相关链接

- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/) 