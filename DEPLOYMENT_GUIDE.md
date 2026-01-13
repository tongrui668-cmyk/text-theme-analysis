# 模型部署指南

## 项目概述

本项目是一个基于LDA（潜在狄利克雷分配）和随机森林分类器的主题分析模型，用于分析用户评论和文本内容的主题分布。

## 部署架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   客户端请求    │────>│  Flask API服务  │────>│  模型预测      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                      │
                                      │
                             ┌───────────────┐
                             │  结果返回    │
                             └───────────────┘
```

## 部署方式

### 方式一：使用部署脚本（推荐）

1. **运行部署脚本**
   ```bash
   python create_deployment.py
   ```
   
   脚本会自动创建`deployment`目录和`model_deployment.zip`压缩包。

2. **解压部署包**
   ```bash
   unzip model_deployment.zip
   cd deployment
   ```

3. **安装依赖**
   ```bash
   # 创建虚拟环境
   python3 -m venv venv
   
   # 激活虚拟环境
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   # 安装依赖
   pip install -r requirements.txt
   ```

4. **启动服务**
   ```bash
   # 开发环境
   python app.py
   
   # 生产环境
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### 方式二：手动部署

1. **准备环境**
   ```bash
   python3 -m venv venv
   # 激活虚拟环境
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```

2. **安装依赖**
   ```bash
   pip install numpy pandas scikit-learn jieba gensim nltk Flask gunicorn
   ```

3. **复制文件**
   - 复制`data/models/`目录下的所有模型文件
   - 复制`src/`目录下的源代码文件
   - 创建`app.py`文件作为Flask应用入口

4. **启动服务**
   ```bash
   python app.py
   ```

## API接口说明

### 1. 健康检查接口
- **URL**: `/api/health`
- **方法**: GET
- **返回示例**:
  ```json
  {
    "status": "healthy",
    "models_loaded": true
  }
  ```

### 2. 主题分析接口
- **URL**: `/api/analyze`
- **方法**: POST
- **请求体**:
  ```json
  {
    "text": "这个APP很好用，推荐给大家"
  }
  ```
- **返回示例**:
  ```json
  {
    "success": true,
    "result": {
      "theme": "用户体验评价",
      "confidence": 0.92,
      "keywords": ["好用", "推荐", "APP"],
      "topic_distribution": [
        {"topic": "用户体验评价", "probability": 0.65},
        {"topic": "功能建议", "probability": 0.15},
        {"topic": "情感表达", "probability": 0.20}
      ]
    }
  }
  ```

## 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| FLASK_ENV | development | 运行环境 |
| FLASK_APP | app.py | 应用入口 |
| MODEL_DIR | ./models | 模型文件目录 |
| PORT | 5000 | 服务端口 |
| HOST | 0.0.0.0 | 服务主机 |

## 性能优化建议

1. **使用gunicorn多进程**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```
   
2. **启用缓存**
   - 对于频繁请求的相同文本，可以启用Redis缓存
   
3. **模型量化**
   - 考虑使用模型量化技术减少模型大小
   
4. **异步处理**
   - 对于大批量文本分析，考虑使用异步处理

## 监控与日志

- Flask默认日志会输出到控制台
- 生产环境建议配置日志文件和监控系统

## 故障排除

### 常见问题

1. **模型文件不存在**
   - 检查`models`目录下是否有所有必要的模型文件
   - 重新运行训练脚本生成模型

2. **依赖安装失败**
   - 确保Python版本为3.8+
   - 尝试使用`pip install --upgrade pip`升级pip

3. **服务启动失败**
   - 检查端口是否被占用
   - 查看日志输出的具体错误信息

### 调试命令

```bash
# 检查模型文件
ls -la models/

# 检查依赖
pip list | grep -E "numpy|pandas|scikit-learn|jieba|gensim|Flask"

# 查看服务状态
ps aux | grep gunicorn
```

## 版本管理

- **v1.0.0**: 初始部署版本
  - 包含LDA主题模型和随机森林分类器
  - 支持基本的主题分析功能
  - 提供RESTful API接口

## 安全建议

1. **API密钥认证**
   - 生产环境建议添加API密钥认证
   
2. **请求频率限制**
   - 防止API滥用
   
3. **输入验证**
   - 对输入文本进行长度和内容验证
   
4. **HTTPS加密**
   - 生产环境使用HTTPS协议

## 扩展性考虑

1. **模型更新**
   - 设计模型版本管理机制
   - 支持在线模型切换
   
2. **负载均衡**
   - 高流量场景考虑使用负载均衡
   
3. **容器化部署**
   - 考虑使用Docker容器化部署
   
4. **自动扩缩容**
   - 基于流量自动调整服务实例数量

## 部署清单

- [ ] 模型文件已准备
- [ ] 依赖已安装
- [ ] 配置文件已设置
- [ ] 服务已启动
- [ ] API测试通过
- [ ] 监控已配置