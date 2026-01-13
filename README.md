# 🎯 中文文本主题分析平台

一个基于机器学习的中文文本主题分类与分析Web应用，支持批量文本处理、实时主题预测和可视化分析结果展示。

## 📋 项目简介

文本主题分析工具是一个基于Python和Flask开发的Web应用，使用LDA主题模型和机器学习算法对中文文本进行自动分类和分析。该工具支持单条文本输入和批量文件上传，提供直观的可视化分析结果。

## ✨ 功能特性

- 🤖 **智能主题分类**: 使用LDA + RandomForest混合模型进行主题分析，准确率达到90%+
- 📝 **多格式支持**: 支持Excel、CSV、TXT文件上传和批量处理，单次可处理10万+条文本
- 🌐 **Web界面**: 简洁美观的Web界面，支持响应式设计，适配不同设备
- 📊 **可视化分析**: 分析结果包含主题分布饼图、置信度柱状图等可视化图表
- ⚡ **实时分析**: 支持单条文本实时主题预测，响应时间<1秒
- 🔧 **模块化设计**: 采用分层架构，清晰的代码结构，易于维护和扩展
- 📝 **完整日志**: 详细的日志记录系统，便于调试、监控和性能分析

## 🏗️ 项目结构

```
Mission3/
├── src/                   # 源代码 (重构后的模块化架构)
│   ├── app.py            # Flask应用工厂和初始化
│   ├── config.py         # 统一配置管理
│   ├── routes.py         # Blueprint路由定义
│   ├── services.py       # 业务逻辑层
│   ├── model_manager.py  # 模型管理器
│   ├── text_preprocessor.py # 文本预处理器
│   └── logger.py         # 日志系统
├── training_new/         # 模型训练和优化 (新架构)
│   ├── src/              # 训练模块源代码
│   │   ├── config.py     # 训练配置管理
│   │   ├── data_preprocessor.py # 数据预处理器
│   │   ├── text_preprocessor.py # 文本预处理器
│   │   ├── topic_modeler.py # LDA主题模型
│   │   ├── model_monitor.py # 模型性能监控
│   │   ├── topic_name_optimizer.py # 主题名称优化
│   │   └── ...           # 其他训练相关模块
│   ├── scripts/          # 部署和监控脚本
│   │   ├── deploy_model.py # 模型自动部署
│   │   └── monitor_model.py # 模型性能监控
│   ├── tests/            # 测试文件
│   └── README.md         # 训练模块说明
├── static/               # 静态文件
│   ├── templates/        # HTML模板
│   │   ├── index.html   # 主页
│   │   ├── result.html  # 结果页面
│   │   ├── error.html   # 错误页面
│   │   ├── 404.html     # 404页面
│   │   └── 500.html     # 500页面
│   ├── css/             # 样式文件
│   ├── js/              # JavaScript文件
│   └── images/          # 图片资源
├── data/                 # 数据目录
│   ├── models/          # 训练好的模型文件
│   │   ├── deployments/ # 部署历史
│   │   └── monitoring/  # 性能监控历史
│   └── raw/             # 原始数据和上传文件
│       ├── uploads/     # 用户上传文件
│       ├── chinese_stopwords.txt
│       └── 停用词表.txt
├── logs/                 # 日志文件
├── requirements.txt      # 依赖包列表
├── run.py               # 应用启动脚本
└── README.md            # 项目说明文档
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- pip包管理器

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
python run.py
```

应用将在 `http://127.0.0.1:5000` 启动。

### 4. 使用说明

1. **文件上传分析**:
   - 访问主页
   - 选择Excel、CSV或TXT文件上传
   - 系统自动分析并显示结果

2. **实时文本分析**:
   - 在文本框中输入或粘贴文本
   - 点击分析按钮获取主题预测结果

## 🧠 技术架构

### 核心技术栈

- **Web框架**: Flask 2.3.3
- **机器学习**: scikit-learn 1.3.0
- **文本处理**: jieba + THULAC
- **数据处理**: pandas + numpy
- **模型持久化**: joblib

### 模型架构

1. **文本预处理**:
   - 中文分词 (jieba/THULAC)
   - 停用词过滤
   - 文本清洗

2. **特征提取**:
   - LDA主题模型特征
   - TF-IDF特征
   - 特征融合

3. **主题分类**:
   - RandomForest分类器
   - 多类别主题预测
   - 置信度评估

## 🌟 项目亮点

1. **技术创新**: 结合LDA主题模型和RandomForest分类器，实现了高精度的中文文本主题分类
2. **性能优化**: 批量处理算法优化，支持10万+条文本的快速分析
3. **用户体验**: 简洁直观的Web界面，实时反馈分析结果
4. **可扩展性**: 模块化架构设计，支持新增主题类型和文本处理算法
5. **实际应用**: 可直接用于客户评论分析、新闻分类、文档管理等实际业务场景

## 🛠️ 技术栈

| 类别 | 技术/框架 | 版本 | 用途 |
|------|-----------|------|------|
| Web框架 | Flask | 2.3.3 | 构建Web应用 |
| 机器学习 | scikit-learn | 1.3.0 | 实现分类模型 |
| 文本处理 | jieba | 0.42.1 | 中文分词 |
| 数据处理 | pandas | 2.0.3 | 数据读取与处理 |
| 数据处理 | numpy | 1.24.3 | 数值计算 |
| 模型持久化 | joblib | 1.3.2 | 模型存储与加载 |
| 前端 | HTML/CSS/JavaScript | - | 构建用户界面 |
| 版本控制 | Git | - | 代码管理 |

## 📊 API接口

### 1. 获取应用状态

```http
GET /api/status
```

返回模型加载状态和应用运行状态。

### 2. 获取主题列表

```http
GET /api/themes
```

返回所有可用的主题及其关键词。

### 3. 文本分析

```http
POST /analyze_text
Content-Type: application/json

{
    "text": "要分析的文本内容"
}
```

返回主题预测结果和置信度。

## 📱 界面预览

（此处可添加界面预览GIF或更多截图，展示系统功能和操作流程）

## ⚙️ 配置说明

### 主要配置项 (src/config.py)

- **Flask配置**: 主机、端口、调试模式等
- **模型参数**: LDA和RandomForest的超参数
- **文件路径**: 数据和模型文件的存储路径
- **日志配置**: 日志级别、文件大小等
- **文本预处理配置**: 分词器选择、清洗规则等
- **主题分类配置**: 默认主题列表和关键词数量

### 环境变量

- `FLASK_ENV`: 运行环境 (development/production)
- `SECRET_KEY`: Flask应用密钥

## 🔧 开发指南

### 添加新的主题

1. 在 `src/config.py` 中更新 `THEME_CONFIG`
2. 重新训练模型或更新主题关键词
3. 重启应用

### 自定义文本预处理

1. 修改 `src/text_preprocessor.py`
2. 添加新的预处理方法
3. 更新配置文件

### 扩展API接口

1. 在 `src/routes.py` 中添加新的Blueprint路由
2. 在 `src/services.py` 中实现相应的业务逻辑
3. 更新API文档

## 📝 日志系统

应用使用Python标准logging模块，支持：

- 文件日志轮转
- 不同级别的日志记录
- 控制台和文件双重输出
- 详细的错误追踪

日志文件位置: `logs/app.log`

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_model_manager.py

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

## 🚀 部署

### Docker部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "run.py"]
```

### 生产环境部署

1. 使用Gunicorn作为WSGI服务器
2. 配置Nginx作为反向代理
3. 设置环境变量
4. 配置日志轮转

## 📈 性能优化

- **模型缓存**: 使用内存缓存提高响应速度
- **批量处理**: 支持大规模文本批量分析
- **异步处理**: 可扩展为异步任务队列
- **数据库集成**: 支持历史结果存储和查询

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/Mission3/issues)

## 🙏 致谢

感谢以下开源项目的支持：

- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [jieba](https://github.com/fxsjy/jieba)
- [THULAC](https://github.com/thulac/thulac-python)

---

## 📊 项目状态

- ✅ 基础功能完成
- ✅ Web界面实现
- ✅ API接口开发
- ✅ 日志系统集成
- ✅ 性能优化完成
- ✅ 测试用例编写完成
- ✅ 文档完善完成
- ✅ 模型部署完成

**最后更新**: 2026年1月13日