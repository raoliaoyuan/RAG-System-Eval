from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from typing import List
import uvicorn
import time
import json

from rag_system import RAGSystem
from qa_system import QASystem
from config import Config
from rag_evaluator import RAGEvaluator

app = FastAPI(title="RAG演示系统", description="检索增强生成系统演示")

# 创建上传目录
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 全局变量存储当前配置
current_config = {
    "llm_provider": "ollama",
    "llm_config": Config.get_ollama_llm_config(),
    "embedding_provider": "ollama", 
    "embedding_config": Config.get_ollama_embedding_config()
}

# 初始化RAG系统、QA系统和评估器
rag_system = RAGSystem(embedding_config=current_config["embedding_config"])
qa_system = QASystem(rag_system, llm_config=current_config["llm_config"])
evaluator = RAGEvaluator()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG演示系统</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { 
                display: flex;
                min-height: 100vh;
                width: 100vw;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 30px rgba(0,0,0,0.1);
            }
            .sidebar {
                width: 280px;
                min-width: 220px;
                max-width: 320px;
                background: #f8f9fa;
                border-right: 1px solid #e9ecef;
                padding: 16px 10px;
                overflow-y: auto;
                font-size: 13px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .sidebar h1 {
                font-size: 1.15em;
                margin-bottom: 12px;
            }
            .sidebar h2 {
                font-size: 1em;
                margin-bottom: 7px;
                padding-bottom: 3px;
            }
            .sidebar .section {
                padding: 8px;
                margin: 10px 0;
            }
            .sidebar label {
                font-size: 0.95em;
            }
            .sidebar button {
                font-size: 12px;
                padding: 8px 12px;
            }
            .main-content {
                flex: 1;
                padding: 18px 16px;
                background: white;
                min-width: 0;
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                margin-bottom: 18px;
                font-size: 1.6em;
                font-weight: 300;
            }
            h2 { 
                color: #34495e; 
                margin-bottom: 10px;
                font-size: 1.1em;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }
            .section { 
                margin: 15px 0; 
                padding: 12px; 
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                border: 1px solid #e9ecef;
            }
            .form-group { margin: 10px 0; }
            input[type="file"], input[type="text"], textarea, select { 
                width: 100%; 
                padding: 8px; 
                border: 2px solid #e9ecef; 
                border-radius: 6px;
                font-size: 13px;
                transition: border-color 0.3s;
            }
            input[type="file"]:focus, input[type="text"]:focus, textarea:focus, select:focus { 
                outline: none;
                border-color: #3498db;
            }
            button { 
                background: linear-gradient(45deg, #3498db, #2980b9); 
                color: white; 
                padding: 10px 18px; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer;
                font-size: 13px;
                font-weight: 600;
                transition: all 0.3s;
                width: 100%;
            }
            button:hover { 
                background: linear-gradient(45deg, #2980b9, #1f5f8b);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            }
            .result { 
                margin-top: 20px; 
                padding: 10px; 
                background: #f8f9fa; 
                border-radius: 6px;
                border-left: 4px solid #3498db;
            }
            .stats { 
                display: grid; 
                grid-template-columns: 1fr; 
                gap: 15px; 
                margin: 20px 0; 
            }
            .stat-card { 
                background: linear-gradient(135deg, #74b9ff, #0984e3); 
                color: white;
                padding: 12px; 
                border-radius: 7px; 
                text-align: center;
                box-shadow: 0 4px 15px rgba(116, 185, 255, 0.3);
            }
            .stat-card h3 { 
                font-size: 0.9em; 
                margin-bottom: 8px;
                opacity: 0.9;
            }
            .stat-card p { 
                font-size: 1.8em; 
                font-weight: bold;
                margin: 0;
            }
            .qa-container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .qa-header {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 25px;
                text-align: center;
            }
            .qa-content {
                padding: 18px 0;
            }
            .question-section {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 7px;
                margin-bottom: 12px;
            }
            .answer-section {
                background: #e8f5e8;
                padding: 12px;
                border-radius: 7px;
                margin-bottom: 12px;
                border-left: 4px solid #27ae60;
            }
            .sources-section {
                background: #fff3cd;
                padding: 12px;
                border-radius: 7px;
                border-left: 4px solid #ffc107;
            }
            .source-item {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            .source-rank {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                margin-right: 10px;
            }
            .source-score {
                color: #e74c3c;
                font-weight: bold;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 8px;
                margin: 10px 0;
            }
            .metric-card {
                background: #ecf0f1;
                padding: 8px;
                border-radius: 6px;
                text-align: center;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }
            .metric-label {
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 5px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #7f8c8d;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #dc3545;
            }
            .success {
                background: #d4edda;
                color: #155724;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #28a745;
            }
            .config-section {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
            }
            .provider-select {
                margin-bottom: 15px;
            }
            .config-fields {
                display: none;
            }
            .config-fields.active {
                display: block;
            }
            .config-row {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }
            .config-row input {
                flex: 1;
            }
            .config-row label {
                min-width: 80px;
                margin-bottom: 0;
                line-height: 44px;
            }
            .test-button {
                background: linear-gradient(45deg, #27ae60, #2ecc71);
                margin-top: 10px;
            }
            .test-button:hover {
                background: linear-gradient(45deg, #2ecc71, #27ae60);
            }
            /* 响应式布局 */
            @media (max-width: 1200px) {
                .container { flex-direction: column; }
                .sidebar {
                    width: 100vw;
                    max-width: 100vw;
                    min-width: 0;
                    border-right: none;
                    border-bottom: 1px solid #e9ecef;
                    padding: 10px 4vw;
                }
                .main-content {
                    padding: 10px 4vw;
                }
            }
            @media (max-width: 800px) {
                .container { flex-direction: column; }
                .sidebar {
                    width: 100vw;
                    max-width: 100vw;
                    min-width: 0;
                    border-right: none;
                    border-bottom: 1px solid #e9ecef;
                    padding: 8px 2vw;
                }
                .main-content {
                    padding: 8px 2vw;
                }
                h1 { font-size: 1.1em; }
            }
            .sidebar .stats {
                gap: 6px;
                margin: 8px 0;
            }
            .sidebar .stat-card {
                padding: 6px 4px;
                border-radius: 5px;
                font-size: 12px;
            }
            .sidebar .stat-card h3 {
                font-size: 0.85em;
                margin-bottom: 3px;
            }
            .sidebar .stat-card p {
                font-size: 1.1em;
                margin: 0;
            }
            .sidebar .metrics {
                gap: 4px;
                margin: 6px 0;
            }
            .sidebar .metric-card {
                padding: 4px;
                border-radius: 4px;
                font-size: 11px;
            }
            .sidebar .metric-value {
                font-size: 1em;
            }
            .sidebar .metric-label {
                font-size: 0.8em;
            }
            .sidebar .config-section {
                padding: 6px 4px;
                margin: 8px 0;
            }
            .sidebar .provider-select {
                margin-bottom: 7px;
            }
            .sidebar .config-row {
                gap: 4px;
                margin-bottom: 5px;
            }
            .sidebar .config-row input,
            .sidebar .config-row select {
                padding: 5px 6px;
                font-size: 12px;
                border-radius: 4px;
            }
            .sidebar .config-row label {
                min-width: 60px;
                font-size: 0.92em;
                line-height: 28px;
            }
            .sidebar .config-fields {
                padding: 2px 0;
            }
            .sidebar .test-button {
                font-size: 11px;
                padding: 6px 10px;
                margin-top: 5px;
            }
            
            /* 评估结果样式 */
            .evaluation-overview {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            
            .overall-score {
                text-align: center;
                margin-bottom: 25px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 10px;
                color: white;
            }
            
            .score-display {
                margin-top: 10px;
            }
            
            .score-value {
                font-size: 3em;
                font-weight: bold;
                display: block;
            }
            
            .score-label {
                font-size: 1.2em;
                opacity: 0.8;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }
            
            .metric-item {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                transition: transform 0.2s;
            }
            
            .metric-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .metric-header {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 0.95em;
            }
            
            .metric-value {
                font-size: 1.8em;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 5px;
            }
            
            .metric-detail {
                font-size: 0.85em;
                color: #7f8c8d;
                line-height: 1.3;
            }
            
            /* 评估分类样式 */
            .evaluation-category {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin: 20px 0;
            }
            
            .evaluation-category h3 {
                color: #2c3e50;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
                font-size: 1.3em;
            }
            
            /* 详细指标样式 */
            .metric-item.detailed {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #3498db;
                transition: all 0.3s ease;
                margin-bottom: 15px;
            }
            
            .metric-item.detailed:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            
            .metric-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 15px;
                font-weight: bold;
                color: #2c3e50;
                font-size: 1.1em;
            }
            
            .metric-icon {
                margin-right: 8px;
                font-size: 1.2em;
            }
            
            .metric-score {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }
            
            /* 指标分解样式 */
            .metric-breakdown {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border: 1px solid #e9ecef;
            }
            
            .breakdown-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
                border-bottom: 1px solid #f1f3f4;
            }
            
            .breakdown-item:last-child {
                border-bottom: none;
            }
            
            .breakdown-label {
                font-weight: 500;
                color: #5a6c7d;
                font-size: 0.9em;
            }
            
            .breakdown-value {
                font-weight: bold;
                color: #2c3e50;
                font-size: 0.9em;
            }
            
            /* 指标描述样式 */
            .metric-description {
                background: #e8f4fd;
                padding: 12px;
                border-radius: 6px;
                border-left: 3px solid #3498db;
                font-size: 0.85em;
                color: #2c3e50;
                line-height: 1.4;
            }
            
            /* 评估详情样式 */
            .evaluation-details {
                background: white;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin-top: 30px;
            }
            
            .evaluation-details h3 {
                color: #2c3e50;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
                font-size: 1.3em;
            }
            
            .details-content {
                display: grid;
                gap: 20px;
            }
            
            .detail-section {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }
            
            .detail-section h4 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.1em;
            }
            
            .detail-section p {
                margin-bottom: 10px;
                line-height: 1.6;
                color: #5a6c7d;
            }
            
            .detail-section strong {
                color: #2c3e50;
            }
            
            /* 总体评分描述 */
            .score-description {
                margin-top: 15px;
                padding: 15px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                font-size: 0.9em;
                line-height: 1.4;
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 左侧边栏 -->
            <div class="sidebar">
                <h1>🔍 RAG系统</h1>
                
                <!-- 模型配置 -->
                <div class="section config-section">
                    <h2>⚙️ 模型配置</h2>
                    
                    <!-- LLM配置 -->
                    <div class="form-group">
                        <label>🤖 LLM提供商:</label>
                        <select id="llm-provider" onchange="toggleLLMConfig()">
                            <option value="ollama">本地 Ollama</option>
                            <option value="openai">OpenAI API</option>
                            <option value="custom">自定义 API</option>
                        </select>
                    </div>
                    
                    <div id="llm-ollama-config" class="config-fields active">
                        <div class="config-row">
                            <label>地址:</label>
                            <input type="text" id="ollama-base-url" value="http://localhost:11434" placeholder="Ollama服务地址">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="ollama-llm-model" value="llama3" placeholder="LLM模型名称">
                        </div>
                    </div>
                    
                    <div id="llm-openai-config" class="config-fields">
                        <div class="config-row">
                            <label>API Key:</label>
                            <input type="text" id="openai-api-key" placeholder="OpenAI API Key">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="openai-model" value="gpt-3.5-turbo" placeholder="OpenAI模型名称">
                        </div>
                        <div class="config-row">
                            <label>Base URL:</label>
                            <input type="text" id="openai-base-url" value="https://api.openai.com/v1" placeholder="OpenAI API Base URL">
                        </div>
                    </div>
                    
                    <div id="llm-custom-config" class="config-fields">
                        <div class="config-row">
                            <label>API URL:</label>
                            <input type="text" id="custom-api-url" placeholder="自定义API地址">
                        </div>
                        <div class="config-row">
                            <label>API Key:</label>
                            <input type="text" id="custom-api-key" placeholder="API Key">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="custom-model" placeholder="模型名称">
                        </div>
                    </div>
                    
                    <button class="test-button" onclick="testLLMConnection()">🧪 测试LLM连接</button>
                    
                    <!-- Embedding配置 -->
                    <div class="form-group" style="margin-top: 25px;">
                        <label>🔤 Embedding提供商:</label>
                        <select id="embedding-provider" onchange="toggleEmbeddingConfig()">
                            <option value="ollama">本地 Ollama</option>
                            <option value="openai">OpenAI API</option>
                            <option value="custom">自定义 API</option>
                        </select>
                    </div>
                    
                    <div id="embedding-ollama-config" class="config-fields active">
                        <div class="config-row">
                            <label>地址:</label>
                            <input type="text" id="ollama-embedding-url" value="http://localhost:11434" placeholder="Ollama服务地址">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="ollama-embedding-model" value="bge-small-zh" placeholder="Embedding模型名称">
                        </div>
                    </div>
                    
                    <div id="embedding-openai-config" class="config-fields">
                        <div class="config-row">
                            <label>API Key:</label>
                            <input type="text" id="openai-embedding-key" placeholder="OpenAI API Key">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="openai-embedding-model" value="text-embedding-ada-002" placeholder="OpenAI Embedding模型">
                        </div>
                        <div class="config-row">
                            <label>Base URL:</label>
                            <input type="text" id="openai-embedding-base-url" value="https://api.openai.com/v1" placeholder="OpenAI API Base URL">
                        </div>
                    </div>
                    
                    <div id="embedding-custom-config" class="config-fields">
                        <div class="config-row">
                            <label>API URL:</label>
                            <input type="text" id="custom-embedding-url" placeholder="自定义Embedding API地址">
                        </div>
                        <div class="config-row">
                            <label>API Key:</label>
                            <input type="text" id="custom-embedding-key" placeholder="API Key">
                        </div>
                        <div class="config-row">
                            <label>模型:</label>
                            <input type="text" id="custom-embedding-model" placeholder="Embedding模型名称">
                        </div>
                    </div>
                    
                    <button class="test-button" onclick="testEmbeddingConnection()">🧪 测试Embedding连接</button>
                    <button onclick="applyConfig()" style="margin-top: 10px;">💾 应用配置</button>
                </div>
                
                <!-- 系统状态 -->
                <div class="section">
                    <h2>📊 系统状态</h2>
                    <div id="stats" class="stats">
                        <div class="stat-card">
                            <h3>文档数量</h3>
                            <p id="doc-count">0</p>
                        </div>
                        <div class="stat-card">
                            <h3>索引状态</h3>
                            <p id="index-status">未初始化</p>
                        </div>
                        <div class="stat-card">
                            <h3>LLM模型</h3>
                            <p id="llm-model">llama3</p>
                        </div>
                        <div class="stat-card">
                            <h3>嵌入模型</h3>
                            <p id="embedding-model">bge-small-zh</p>
                        </div>
                    </div>
                    <button onclick="loadStats()">🔄 刷新状态</button>
                </div>

                <!-- 文档上传 -->
                <div class="section">
                    <h2>📁 文档上传</h2>
                    <form id="upload-form" enctype="multipart/form-data">
                        <div class="form-group">
                            <label>选择文档 (支持PDF, DOCX, TXT):</label>
                            <input type="file" name="files" multiple accept=".pdf,.docx,.doc,.txt">
                        </div>
                        <button type="submit">📤 上传文档</button>
                    </form>
                    <div id="upload-result"></div>
                </div>

                <!-- 构建索引 -->
                <div class="section">
                    <h2>🔧 构建索引</h2>
                    <button onclick="buildIndex()">⚡ 构建向量索引</button>
                    <div id="index-result"></div>
                </div>
            </div>

            <!-- 右侧主内容区 -->
            <div class="main-content">
                <div class="qa-container">
                    <div class="qa-header">
                        <h1>❓ 智能问答系统</h1>
                        <p>基于检索增强生成的智能问答</p>
                    </div>
                    
                    <div class="qa-content">
                        <!-- 问题输入 -->
                        <div class="question-section">
                            <h2>💭 输入您的问题</h2>
                            <div class="form-group">
                                <textarea id="query" rows="4" placeholder="请输入您的问题，系统将基于已上传的文档进行回答..."></textarea>
                            </div>
                            <button onclick="askQuestion()">🚀 开始提问</button>
                            <button onclick="evaluateResponse()" style="margin-left: 10px; background: linear-gradient(45deg, #27ae60, #2ecc71);">📊 评估回答</button>
                        </div>

                        <!-- 答案展示 -->
                        <div id="qa-result"></div>
                        
                        <!-- 评估结果展示 -->
                        <div id="evaluation-result" style="display: none;">
                            <h2>📊 系统评估结果</h2>
                            <div id="evaluation-content"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function toggleLLMConfig() {
                const provider = document.getElementById('llm-provider').value;
                document.querySelectorAll('#llm-ollama-config, #llm-openai-config, #llm-custom-config').forEach(el => {
                    el.classList.remove('active');
                });
                document.getElementById(`llm-${provider}-config`).classList.add('active');
            }
            
            function toggleEmbeddingConfig() {
                const provider = document.getElementById('embedding-provider').value;
                document.querySelectorAll('#embedding-ollama-config, #embedding-openai-config, #embedding-custom-config').forEach(el => {
                    el.classList.remove('active');
                });
                document.getElementById(`embedding-${provider}-config`).classList.add('active');
            }
            
            async function testLLMConnection() {
                const provider = document.getElementById('llm-provider').value;
                let config = {};
                
                if (provider === 'ollama') {
                    config = {
                        provider: 'ollama',
                        base_url: document.getElementById('ollama-base-url').value,
                        model: document.getElementById('ollama-llm-model').value
                    };
                } else if (provider === 'openai') {
                    config = {
                        provider: 'openai',
                        api_key: document.getElementById('openai-api-key').value,
                        model: document.getElementById('openai-model').value,
                        base_url: document.getElementById('openai-base-url').value
                    };
                } else if (provider === 'custom') {
                    config = {
                        provider: 'custom',
                        api_url: document.getElementById('custom-api-url').value,
                        api_key: document.getElementById('custom-api-key').value,
                        model: document.getElementById('custom-model').value
                    };
                }
                
                try {
                    const response = await fetch('/test-llm', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });
                    const result = await response.json();
                    alert(result.success ? '✅ LLM连接测试成功！' : '❌ LLM连接测试失败：' + result.error);
                } catch (error) {
                    alert('❌ 测试失败：' + error);
                }
            }
            
            async function testEmbeddingConnection() {
                const provider = document.getElementById('embedding-provider').value;
                let config = {};
                
                if (provider === 'ollama') {
                    config = {
                        provider: 'ollama',
                        base_url: document.getElementById('ollama-embedding-url').value,
                        model: document.getElementById('ollama-embedding-model').value
                    };
                } else if (provider === 'openai') {
                    config = {
                        provider: 'openai',
                        api_key: document.getElementById('openai-embedding-key').value,
                        model: document.getElementById('openai-embedding-model').value,
                        base_url: document.getElementById('openai-embedding-base-url').value
                    };
                } else if (provider === 'custom') {
                    config = {
                        provider: 'custom',
                        api_url: document.getElementById('custom-embedding-url').value,
                        api_key: document.getElementById('custom-embedding-key').value,
                        model: document.getElementById('custom-embedding-model').value
                    };
                }
                
                try {
                    const response = await fetch('/test-embedding', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });
                    const result = await response.json();
                    alert(result.success ? '✅ Embedding连接测试成功！' : '❌ Embedding连接测试失败：' + result.error);
                } catch (error) {
                    alert('❌ 测试失败：' + error);
                }
            }
            
            async function applyConfig() {
                const llmProvider = document.getElementById('llm-provider').value;
                const embeddingProvider = document.getElementById('embedding-provider').value;
                
                let llmConfig = {};
                let embeddingConfig = {};
                
                // 构建LLM配置
                if (llmProvider === 'ollama') {
                    llmConfig = {
                        provider: 'ollama',
                        base_url: document.getElementById('ollama-base-url').value,
                        model: document.getElementById('ollama-llm-model').value
                    };
                } else if (llmProvider === 'openai') {
                    llmConfig = {
                        provider: 'openai',
                        api_key: document.getElementById('openai-api-key').value,
                        model: document.getElementById('openai-model').value,
                        base_url: document.getElementById('openai-base-url').value
                    };
                } else if (llmProvider === 'custom') {
                    llmConfig = {
                        provider: 'custom',
                        api_url: document.getElementById('custom-api-url').value,
                        api_key: document.getElementById('custom-api-key').value,
                        model: document.getElementById('custom-model').value
                    };
                }
                
                // 构建Embedding配置
                if (embeddingProvider === 'ollama') {
                    embeddingConfig = {
                        provider: 'ollama',
                        base_url: document.getElementById('ollama-embedding-url').value,
                        model: document.getElementById('ollama-embedding-model').value
                    };
                } else if (embeddingProvider === 'openai') {
                    embeddingConfig = {
                        provider: 'openai',
                        api_key: document.getElementById('openai-embedding-key').value,
                        model: document.getElementById('openai-embedding-model').value,
                        base_url: document.getElementById('openai-embedding-base-url').value
                    };
                } else if (embeddingProvider === 'custom') {
                    embeddingConfig = {
                        provider: 'custom',
                        api_url: document.getElementById('custom-embedding-url').value,
                        api_key: document.getElementById('custom-embedding-key').value,
                        model: document.getElementById('custom-embedding-model').value
                    };
                }
                
                try {
                    const response = await fetch('/apply-config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            llm_config: llmConfig,
                            embedding_config: embeddingConfig
                        })
                    });
                    const result = await response.json();
                    if (result.success) {
                        alert('✅ 配置应用成功！');
                        loadStats();
                    } else {
                        alert('❌ 配置应用失败：' + result.error);
                    }
                } catch (error) {
                    alert('❌ 应用配置失败：' + error);
                }
            }

            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    document.getElementById('doc-count').textContent = stats.document_count;
                    document.getElementById('index-status').textContent = stats.is_initialized ? '✅ 已初始化' : '❌ 未初始化';
                    document.getElementById('embedding-model').textContent = stats.model_name || 'bge-small-zh';
                    document.getElementById('llm-model').textContent = stats.llm_model || 'unknown';
                } catch (error) {
                    console.error('加载状态失败:', error);
                }
            }

            async function uploadFiles() {
                const formData = new FormData();
                const fileInput = document.querySelector('input[type="file"]');
                
                for (let file of fileInput.files) {
                    formData.append('files', file);
                }

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('upload-result').innerHTML = `<div class="result success">✅ ${result.message}</div>`;
                    loadStats();
                } catch (error) {
                    document.getElementById('upload-result').innerHTML = `<div class="result error">❌ 上传失败: ${error}</div>`;
                }
            }

            async function buildIndex() {
                try {
                    const response = await fetch('/build-index', { method: 'POST' });
                    const result = await response.json();
                    document.getElementById('index-result').innerHTML = `<div class="result success">✅ ${result.message}</div>`;
                    loadStats();
                } catch (error) {
                    document.getElementById('index-result').innerHTML = `<div class="result error">❌ 构建索引失败: ${error}</div>`;
                }
            }

            async function askQuestion() {
                const query = document.getElementById('query').value;
                if (!query.trim()) {
                    alert('请输入问题');
                    return;
                }

                // 显示加载状态
                document.getElementById('qa-result').innerHTML = '<div class="loading">🤔 正在思考中，请稍候...</div>';

                try {
                    const startTime = Date.now();
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    const result = await response.json();
                    const endTime = Date.now();
                    const responseTime = endTime - startTime;
                    
                    // 存储最新的问答结果用于评估
                    lastQAResult = {
                        query: query,
                        answer: result.answer,
                        sources: result.sources,
                        response_time: responseTime / 1000.0  // 转换为秒
                    };
                    
                    let html = '';
                    if (result.error) {
                        html = `<div class="answer-section error">❌ 错误: ${result.error}</div>`;
                    } else {
                        // 系统指标
                        html += `<div class="metrics">
                            <div class="metric-card">
                                <div class="metric-value">${responseTime}ms</div>
                                <div class="metric-label">响应时间</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${(result.confidence * 100).toFixed(1)}%</div>
                                <div class="metric-label">置信度</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${result.sources.length}</div>
                                <div class="metric-label">来源文档</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">${result.answer.length}</div>
                                <div class="metric-label">答案长度</div>
                            </div>
                        </div>`;

                        // 问题
                        html += `<div class="question-section">
                            <h3>❓ 问题</h3>
                            <p>${result.query}</p>
                        </div>`;

                        // 答案
                        html += `<div class="answer-section">
                            <h3>💡 答案</h3>
                            <p>${result.answer}</p>
                        </div>`;

                        // 来源文档
                        html += `<div class="sources-section">
                            <h3>📚 来源文档 (${result.sources.length}个)</h3>`;
                        result.sources.forEach((source, index) => {
                            html += `<div class="source-item">
                                <div>
                                    <span class="source-rank">#${source.rank}</span>
                                    <span class="source-score">相似度: ${(source.score * 100).toFixed(1)}%</span>
                                </div>
                                <p style="margin-top: 10px;">${source.content}</p>
                            </div>`;
                        });
                        html += `</div>`;
                    }
                    document.getElementById('qa-result').innerHTML = html;
                } catch (error) {
                    document.getElementById('qa-result').innerHTML = `<div class="answer-section error">❌ 提问失败: ${error}</div>`;
                }
            }

            // 绑定表单提交事件
            document.getElementById('upload-form').addEventListener('submit', function(e) {
                e.preventDefault();
                uploadFiles();
            });

            // 页面加载时获取状态
            loadStats();
            
            // 全局变量存储最新的问答结果
            let lastQAResult = null;
            
            async function evaluateResponse() {
                if (!lastQAResult) {
                    alert('请先提问获取回答，然后再进行评估');
                    return;
                }
                
                // 显示加载状态
                document.getElementById('evaluation-content').innerHTML = '<div class="loading">🔍 正在评估中，请稍候...</div>';
                document.getElementById('evaluation-result').style.display = 'block';
                
                try {
                    const response = await fetch('/evaluate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: lastQAResult.query,
                            answer: lastQAResult.answer,
                            retrieved_sources: lastQAResult.sources,
                            response_time: lastQAResult.response_time
                        })
                    });
                    
                    const evaluation = await response.json();
                    
                    if (evaluation.error) {
                        document.getElementById('evaluation-content').innerHTML = `<div class="error">❌ 评估失败: ${evaluation.error}</div>`;
                    } else {
                        displayEvaluationResults(evaluation);
                    }
                } catch (error) {
                    document.getElementById('evaluation-content').innerHTML = `<div class="error">❌ 评估失败: ${error}</div>`;
                }
            }
            
            function displayEvaluationResults(evaluation) {
                const overall = evaluation.overall_score;
                const relevance = evaluation.answer_relevance;
                const faithfulness = evaluation.answer_faithfulness;
                const precision = evaluation.context_precision;
                const recall = evaluation.context_recall;
                const completeness = evaluation.answer_completeness;
                const consistency = evaluation.answer_consistency;
                const diversity = evaluation.source_diversity;
                const performance = evaluation.performance_metrics;
                
                let html = `
                    <div class="evaluation-overview">
                        <!-- 总体评分 -->
                        <div class="overall-score">
                            <h3>🎯 总体评分</h3>
                            <div class="score-display">
                                <span class="score-value">${(overall.overall_score * 100).toFixed(1)}</span>
                                <span class="score-label">/ 100</span>
                            </div>
                            <div class="score-description">
                                基于8个核心指标的加权平均，权重配置：答案相关性(25%)、忠实度(20%)、精确率(15%)、召回率(15%)、完整性(10%)、一致性(10%)、多样性(5%)
                            </div>
                        </div>
                        
                        <!-- 答案质量评估 -->
                        <div class="evaluation-category">
                            <h3>📝 答案质量评估</h3>
                            <div class="metrics-grid">
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">🎯</span>
                                        答案相关性
                                        <span class="metric-score">${(relevance.overall_relevance * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">关键词重叠度:</span>
                                            <span class="breakdown-value">${(relevance.keyword_overlap * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">长度相关性:</span>
                                            <span class="breakdown-value">${(relevance.keyword_overlap * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：(关键词重叠度 + 长度相关性) / 2<br>
                                        评估答案与查询的语义匹配程度
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">🔗</span>
                                        答案忠实度
                                        <span class="metric-score">${(faithfulness.faithfulness_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">源文档覆盖率:</span>
                                            <span class="breakdown-value">${(faithfulness.source_coverage * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">最大相似度:</span>
                                            <span class="breakdown-value">${(faithfulness.max_source_similarity * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：答案关键词在源文档中的覆盖率<br>
                                        评估答案对源文档的依赖程度和准确性
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">📋</span>
                                        答案完整性
                                        <span class="metric-score">${(completeness.completeness_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">查询类型:</span>
                                            <span class="breakdown-value">${completeness.query_type}</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">答案长度:</span>
                                            <span class="breakdown-value">${completeness.answer_length}字符</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        根据查询类型（事实性/比较性/程序性/一般性）评估完整性<br>
                                        检查是否包含必要的信息要素
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">✅</span>
                                        答案一致性
                                        <span class="metric-score">${(consistency.consistency_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">检查次数:</span>
                                            <span class="breakdown-value">${consistency.consistency_checks}</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">矛盾数量:</span>
                                            <span class="breakdown-value">${consistency.contradictions_count}</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：1 - (矛盾数量 / 检查次数)<br>
                                        评估答案与源文档的信息一致性
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- 检索质量评估 -->
                        <div class="evaluation-category">
                            <h3>🔍 检索质量评估</h3>
                            <div class="metrics-grid">
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">🎯</span>
                                        上下文精确率
                                        <span class="metric-score">${(precision.precision_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">相关文档:</span>
                                            <span class="breakdown-value">${precision.relevant_sources_count}/${precision.total_sources_count}</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">平均相似度:</span>
                                            <span class="breakdown-value">${(precision.avg_similarity * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：相关文档数 / 总文档数<br>
                                        评估检索到的文档与查询的相关程度
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">📚</span>
                                        上下文召回率
                                        <span class="metric-score">${(recall.recall_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">关键词覆盖:</span>
                                            <span class="breakdown-value">${(recall.keyword_coverage * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">覆盖率估计:</span>
                                            <span class="breakdown-value">${(recall.coverage_estimate * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：查询关键词在检索文档中的覆盖率<br>
                                        评估检索系统找到相关信息的能力
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">🌐</span>
                                        源文档多样性
                                        <span class="metric-score">${(diversity.diversity_score * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">唯一文档:</span>
                                            <span class="breakdown-value">${diversity.unique_sources}</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">平均相似度:</span>
                                            <span class="breakdown-value">${(diversity.avg_source_similarity * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        计算方法：1 - 源文档间平均相似度<br>
                                        评估检索结果的多样性和覆盖面
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- 性能指标 -->
                        <div class="evaluation-category">
                            <h3>⚡ 性能指标</h3>
                            <div class="metrics-grid">
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">⏱️</span>
                                        响应时间
                                        <span class="metric-score">${performance.response_time_seconds.toFixed(2)}s</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">处理速度:</span>
                                            <span class="breakdown-value">${performance.tokens_per_second.toFixed(1)}字符/秒</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">查询长度:</span>
                                            <span class="breakdown-value">${performance.query_length}字符</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        从查询提交到获得完整答案的总时间<br>
                                        包括检索、生成和评估的时间
                                    </div>
                                </div>
                                
                                <div class="metric-item detailed">
                                    <div class="metric-header">
                                        <span class="metric-icon">📊</span>
                                        输出统计
                                        <span class="metric-score">${performance.answer_length}字符</span>
                                    </div>
                                    <div class="metric-breakdown">
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">源文档数:</span>
                                            <span class="breakdown-value">${performance.sources_count}个</span>
                                        </div>
                                        <div class="breakdown-item">
                                            <span class="breakdown-label">平均相似度:</span>
                                            <span class="breakdown-value">${(performance.avg_source_score * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                    <div class="metric-description">
                                        答案长度和使用的源文档数量<br>
                                        反映系统的信息密度和检索广度
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- 评估详情 -->
                        <div class="evaluation-details">
                            <h3>📋 评估详情说明</h3>
                            <div class="details-content">
                                <div class="detail-section">
                                    <h4>🎯 答案质量评估</h4>
                                    <p><strong>答案相关性：</strong>评估答案与用户查询的匹配程度，包括关键词重叠和语义相似度。</p>
                                    <p><strong>答案忠实度：</strong>衡量答案对源文档的依赖程度，确保信息准确性。</p>
                                    <p><strong>答案完整性：</strong>根据查询类型评估答案是否包含必要信息要素。</p>
                                    <p><strong>答案一致性：</strong>检查答案与源文档的信息一致性，避免矛盾。</p>
                                </div>
                                
                                <div class="detail-section">
                                    <h4>🔍 检索质量评估</h4>
                                    <p><strong>上下文精确率：</strong>评估检索到的文档与查询的相关程度。</p>
                                    <p><strong>上下文召回率：</strong>衡量检索系统找到相关信息的能力。</p>
                                    <p><strong>源文档多样性：</strong>评估检索结果的多样性和覆盖面。</p>
                                </div>
                                
                                <div class="detail-section">
                                    <h4>⚡ 性能指标</h4>
                                    <p><strong>响应时间：</strong>系统处理查询的总时间，包括检索和生成。</p>
                                    <p><strong>输出统计：</strong>答案长度、源文档数量等统计信息。</p>
                                </div>
                                
                                <div class="detail-section">
                                    <h4>📊 评分标准</h4>
                                    <p><strong>优秀 (80-100%)：</strong>答案准确、完整、相关，检索质量高</p>
                                    <p><strong>良好 (60-80%)：</strong>答案基本满足需求，检索质量较好</p>
                                    <p><strong>一般 (40-60%)：</strong>答案部分满足需求，检索质量一般</p>
                                    <p><strong>较差 (0-40%)：</strong>答案质量较低，检索效果不佳</p>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('evaluation-content').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

@app.get("/stats")
async def get_stats():
    """获取系统状态"""
    stats = rag_system.get_stats()
    # 添加LLM模型信息
    stats['llm_model'] = current_config['llm_config'].get('model', 'unknown')
    stats['llm_provider'] = current_config['llm_config'].get('provider', 'unknown')
    return stats

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传文档"""
    uploaded_files = []
    
    for file in files:
        if file.filename:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file_path)
    
    if uploaded_files:
        rag_system.add_documents(uploaded_files)
        return {"message": f"成功上传 {len(uploaded_files)} 个文件"}
    else:
        raise HTTPException(status_code=400, detail="没有文件被上传")

@app.post("/build-index")
async def build_index():
    """构建索引"""
    try:
        rag_system.build_index()
        return {"message": "索引构建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: dict):
    """提问"""
    try:
        result = qa_system.get_answer_with_sources(query["query"])
        return result
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-llm")
async def test_llm_connection(config: dict):
    """测试LLM连接"""
    try:
        # 这里可以添加实际的LLM连接测试逻辑
        return {"success": True, "message": "LLM连接测试成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/test-embedding")
async def test_embedding_connection(config: dict):
    """测试Embedding连接"""
    try:
        # 这里可以添加实际的Embedding连接测试逻辑
        return {"success": True, "message": "Embedding连接测试成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/apply-config")
async def apply_config(config: dict):
    """应用配置"""
    try:
        global current_config, rag_system, qa_system
        
        # 更新配置
        current_config["llm_config"] = config["llm_config"]
        current_config["embedding_config"] = config["embedding_config"]
        
        # 重新初始化系统
        rag_system = RAGSystem(embedding_config=current_config["embedding_config"])
        qa_system = QASystem(rag_system, llm_config=current_config["llm_config"])
        
        return {"success": True, "message": "配置应用成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/evaluate")
async def evaluate_response(evaluation_data: dict):
    """评估RAG响应"""
    try:
        # 执行评估
        response_time = evaluation_data.get("response_time")
        if response_time is not None:
            response_time = float(response_time)
        else:
            response_time = None
            
        evaluation_results = evaluator.evaluate_rag_response(
            query=evaluation_data["query"],
            answer=evaluation_data["answer"],
            retrieved_sources=evaluation_data["retrieved_sources"],
            response_time=response_time
        )
        
        return evaluation_results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 