# 文本主题分析工具

一个基于机器学习的中文文本主题分析系统，支持文本分类、关键词提取和可视化分析。

## 📋 项目简介

文本主题分析工具是一个基于Python和Flask开发的Web应用，使用LDA主题模型和机器学习算法对中文文本进行自动分类和分析。该工具支持单条文本输入和批量文件上传，提供直观的可视化分析结果。

## ✨ 功能特性

- **中文文本分析**: 支持对中文文本进行主题分类和关键词提取
- **批量处理**: 支持上传TXT/CSV文件进行批量文本分析
- **实时分析**: 提供文本输入框，实时返回主题分类结果和置信度
- **可视化分析结果**: 直观展示主题分布、关键词统计和示例文本分析
- **高性能处理**: 采用并行计算技术，响应时间<1秒
- **高准确率**: 基于LDA主题模型和机器学习分类器，准确率达90%+ 
- **响应式设计**: 适配桌面、平板和移动设备

## 🏗️ 技术栈

| 技术/框架 | 版本 | 用途 |
|---------|------|------|
| Python | 3.8+ | 后端开发语言 |
| Flask | 2.0+ | Web应用框架 |
| Scikit-learn | 1.3+ | 机器学习算法实现 |
| LDA | - | 主题模型 |
| jieba | 0.42+ | 中文分词 |
| pandas | 1.5+ | 数据处理与分析 |
| matplotlib | 3.7+ | 数据可视化 |
| Bootstrap | 5.0+ | 前端UI框架 |
| jQuery | 3.6+ | 前端交互处理 |

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- pip 包管理工具

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/tongrui668-cmyk/text-theme-analysis.git
cd text-theme-analysis
```

2. **创建虚拟环境**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **启动应用**

```bash
python run.py
```

5. **访问应用**

打开浏览器访问: `http://127.0.0.1:5000`

## 📁 项目结构

```
├── src/                  # 主源码目录
│   ├── app.py           # 应用入口
│   ├── routes.py        # 路由定义
│   ├── services.py      # 业务逻辑
│   ├── model_manager.py # 模型管理
│   ├── text_preprocessor.py # 文本预处理
│   ├── logger.py        # 日志配置
│   └── config.py        # 配置文件
├── static/              # 静态资源
│   ├── css/             # CSS样式
│   ├── js/              # JavaScript代码
│   ├── images/          # 项目图片
│   └── templates/       # HTML模板
├── data/                # 数据目录
│   └── models/          # 预训练模型文件
├── training/            # 模型训练目录
│   ├── train_lda_model.py # LDA模型训练脚本
│   ├── train_theme_model.py # 主题分类模型训练脚本
│   ├── data_preprocessor.py # 数据预处理脚本
│   ├── custom_dict.txt  # 自定义词典
│   └── train_log.txt    # 训练日志
├── logs/                # 日志目录
├── requirements.txt     # 依赖列表
├── run.py               # 项目启动文件
├── .gitignore           # Git忽略文件
└── LICENSE              # 许可证文件
```

## 📊 预训练模型

项目提供了完整的预训练模型，用户下载后无需重新训练即可直接使用。所有模型文件位于 `data/models/` 目录下：

| 模型文件 | 用途 |
|---------|------|
| `lda_model.pkl` | LDA主题模型，用于文本主题提取 |
| `theme_classification_model.pkl` | 主题分类器，用于文本主题分类 |
| `vectorizer.pkl` | TF-IDF向量转换器，用于文本特征提取 |
| `count_vectorizer.pkl` | 词频向量转换器，用于文本向量化 |
| `theme_keywords.pkl` | 主题关键词映射表 |

## 🛠️ 模型训练

如果您需要重新训练模型或使用自己的数据集，可以使用 `training/` 目录下的训练脚本：

### 训练文件说明

`training/` 目录包含所有用于模型训练的脚本和资源文件：

| 文件名 | 用途 |
|-------|------|
| `train_lda_model.py` | LDA主题模型训练脚本 |
| `train_theme_model.py` | 主题分类模型训练脚本 |
| `data_preprocessor.py` | 数据预处理脚本 |
| `lad_analyze.py` | LDA主题分析辅助脚本 |
| `pretread.py` | 文本预处理辅助脚本 |
| `custom_dict.txt` | 自定义词典，用于优化分词结果 |
| `train_log.txt` | 训练过程日志记录 |

### 训练步骤

1. **数据准备**
   - 将训练数据放在 `training/` 目录下
   - 数据格式：每行一条文本
   - 支持TXT和CSV格式

2. **预处理数据**
   ```bash
   python training/data_preprocessor.py
   ```

3. **训练LDA主题模型**
   ```bash
   python training/train_lda_model.py
   ```

4. **训练主题分类模型**
   ```bash
   python training/train_theme_model.py
   ```

5. **查看训练日志**
   ```bash
   # Windows
   type training/train_log.txt
   
   # Linux/macOS
   cat training/train_log.txt
   ```

训练完成后，新的模型文件将自动保存到 `data/models/` 目录下，覆盖原有的预训练模型。

## ⚙️ 配置

应用配置文件位于 `src/config.py`，可以修改以下参数：

- **应用配置**: 端口、调试模式
- **模型配置**: 模型文件路径、主题数量
- **文本处理**: 分词器配置、停用词列表
- **日志配置**: 日志级别、日志文件路径

## 🔧 部署

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

推荐使用:
- **Gunicorn**: WSGI服务器
- **Nginx**: 反向代理
- **Supervisor**: 进程管理

## 📝 日志系统

应用使用Python标准logging模块，支持:

- 文件日志轮转
- 不同级别的日志记录
- 控制台和文件双重输出
- 详细的错误追踪

日志文件位置: `logs/app.log`

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
- 🐛 Issues: [GitHub Issues](https://github.com/tongrui668-cmyk/text-theme-analysis/issues)

## 🙏 致谢

感谢以下开源项目的支持：

- [jieba](https://github.com/fxsjy/jieba) - 中文分词库
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Flask](https://flask.palletsprojects.com/) - Web应用框架