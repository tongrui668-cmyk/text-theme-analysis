#!/usr/bin/env python3
"""
模型部署脚本
用于打包模型和依赖，以便在生产环境中部署
"""

import os
import shutil
import zipfile
import subprocess
import sys

def create_deployment_package():
    """创建部署包"""
    # 定义部署目录
    deployment_dir = "deployment"
    model_dir = os.path.join(deployment_dir, "models")
    src_dir = os.path.join(deployment_dir, "src")
    
    # 创建目录结构
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    
    # 复制模型文件
    models_to_copy = [
        "count_vectorizer.pkl",
        "lda_model.pkl", 
        "theme_classification_model.pkl",
        "theme_keywords.pkl",
        "vectorizer.pkl"
    ]
    
    for model_file in models_to_copy:
        src_path = os.path.join("data", "models", model_file)
        dest_path = os.path.join(model_dir, model_file)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"复制模型文件: {model_file}")
        else:
            print(f"警告: 模型文件 {model_file} 不存在")
    
    # 复制源代码文件
    src_files_to_copy = [
        "config.py",
        "text_preprocessor.py",
        "model_manager.py",
        "services.py",
        "logger.py"
    ]
    
    for src_file in src_files_to_copy:
        src_path = os.path.join("src", src_file)
        dest_path = os.path.join(src_dir, src_file)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"复制源代码文件: {src_file}")
        else:
            print(f"警告: 源代码文件 {src_file} 不存在")
    
    # 创建依赖文件
    with open(os.path.join(deployment_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("""
# 核心依赖
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.5.2
jieba==0.42.1
gensim==4.3.3
nltk==3.8.1

# 可选依赖（用于THULAC分词）
# thulac==0.2.1

# 部署相关
Flask==2.0.1
gunicorn==20.1.0
""")
    
    # 创建使用说明
    with open(os.path.join(deployment_dir, "DEPLOYMENT.md"), "w", encoding="utf-8") as f:
        f.write("""# 模型部署说明

## 部署步骤

1. **准备环境**
   ```bash
   # 创建虚拟环境
   python3 -m venv venv
   
   # 激活虚拟环境
   # Windows
   venv\\Scripts\\activate
   # Linux/Mac
   source venv/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **启动服务**
   ```bash
   # 开发环境
   python app.py
   
   # 生产环境
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **测试API**
   ```bash
   # 发送测试请求
   curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "这个APP很好用，推荐给大家"}'
   ```

## 目录结构

```
deployment/
├── models/              # 模型文件
├── src/                 # 源代码
├── app.py               # Flask应用
├── requirements.txt     # 依赖文件
└── DEPLOYMENT.md        # 部署说明
```

## 环境变量

| 变量名 | 默认值 | 描述 |
|-------|-------|------|
| FLASK_ENV | development | 运行环境 |
| FLASK_APP | app.py | 应用入口 |
| MODEL_DIR | ./models | 模型文件目录 |
| PORT | 5000 | 服务端口 |
| HOST | 0.0.0.0 | 服务主机 |
""")
    
    # 创建Flask应用
    with open(os.path.join(deployment_dir, "app.py"), "w", encoding="utf-8") as f:
        f.write("""from flask import Flask, request, jsonify
import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_manager import ModelManager
from text_preprocessor import get_preprocessor

app = Flask(__name__)

# 初始化模型管理器
model_dir = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
model_manager = ModelManager(model_dir=model_dir)

# 初始化预处理器
preprocessor = get_preprocessor()

@app.route("/")
def index():
    return jsonify({
        "message": "主题分析模型API",
        "version": "1.0.0",
        "endpoints": [
            "/api/analyze - 分析文本主题",
            "/api/health - 健康检查"
        ]
    })

@app.route("/api/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": model_manager.models_loaded
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({
                "error": "缺少text参数"
            }), 400
        
        text = data["text"]
        
        # 预处理文本
        preprocessed_text = preprocessor.preprocess(text)
        
        # 分析主题
        result = model_manager.analyze_theme(text)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        app.logger.error(f"分析失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    
    app.run(host=host, port=port, debug=True)
""")
    
    # 创建压缩包
    zip_file = "model_deployment.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # 添加部署目录下的所有文件
        for root, dirs, files in os.walk(deployment_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(deployment_dir))
                zf.write(file_path, arcname)
    
    print(f"创建部署包: {zip_file}")
    print("部署包创建完成！")

if __name__ == "__main__":
    create_deployment_package()