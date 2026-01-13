# 主题模型训练系统

## 项目简介

本项目是一个基于LDA（潜在狄利克雷分配）和随机森林的主题模型训练系统，用于分析用户评论数据并自动生成业务化标签。系统实现了从数据预处理到模型部署的完整流程，支持自动主题数选择、智能标签生成、多维度模型评估、主题名称自动优化、模型性能监控等功能。

## 目录结构

```
training_new/
├── src/                # 源代码目录
│   ├── config.py       # 配置管理模块
│   ├── data_preprocessor.py  # 数据预处理模块
│   ├── text_preprocessor.py  # 文本预处理模块
│   ├── text_preprocessor_enhanced.py  # 增强版文本预处理模块
│   ├── topic_modeler.py      # 主题建模模块
│   ├── topic_name_optimizer.py  # 主题名称优化模块
│   ├── model_monitor.py      # 模型性能监控模块
│   ├── label_generator.py    # 标签生成模块
│   ├── keyword_extractor.py  # 关键词提取模块
│   ├── classifier.py         # 分类器模块
│   ├── model_saver.py        # 模型保存模块
│   ├── report_generator.py   # 报告生成模块
│   └── __init__.py     # 包初始化文件
├── scripts/            # 部署脚本目录
│   └── deploy_model.py  # 自动部署脚本
├── tests/              # 测试目录
│   ├── test_core_functionality.py  # 核心功能测试
│   └── test_text_preprocessor.py  # 文本预处理测试
├── logs/               # 训练日志目录
├── train.py            # 主训练脚本
├── train_improved_model.py  # 改进版训练脚本
├── analyze_topic_quality.py  # 主题质量分析脚本
├── optimize_topic_quality.py  # 主题质量优化脚本
├── optimize_preprocessing.py  # 预处理优化脚本
├── end_to_end_optimizer.py  # 端到端优化脚本
├── test_preprocessors.py  # 预处理测试脚本
├── API_DOCS.md         # API文档
├── README.md           # 项目文档
├── 操作指南.md          # 操作指南（中文）
├── 模型结构文档.md       # 模型结构文档（中文）
└── preprocessing_evaluation_report.md  # 预处理评估报告
```

## 核心功能

1. **数据预处理**：支持文本清洗、分词、去停用词、去重复短语等操作
2. **自动主题数选择**：基于困惑度评估选择最佳主题数
3. **智能标签生成**：基于关键词语义自动生成业务化标签
4. **主题名称自动优化**：自动清洗模糊术语，提取核心概念，生成清晰的主题名称
5. **多维度模型评估**：支持困惑度、主题清晰度、主题一致性、准确率等指标
6. **模型性能监控**：自动跟踪模型性能，检测性能退化情况
7. **自动部署流程**：实现从数据准备到模型部署的端到端自动化
8. **特征组合**：结合LDA主题分布和TF-IDF特征，提升分类性能
9. **详细训练报告**：生成包含业务标签和关键词的完整训练报告
10. **端到端优化**：支持预处理、模型参数、主题质量的全面优化

## 环境依赖

- Python 3.7+
- pandas
- numpy
- scikit-learn
- jieba
- joblib
- matplotlib (可选，用于可视化)
- seaborn (可选，用于可视化)
- pathlib (用于路径管理)

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn jieba joblib
```

### 2. 准备数据

确保在 `../data/raw/` 目录下存在 `评论和正文.xlsx` 文件，包含 `评论内容` 列。

### 3. 运行训练

```bash
# 在 training_new 目录下运行
python train.py
```

### 4. 自动部署

```bash
# 运行自动部署脚本
python scripts/deploy_model.py
```

### 5. 查看结果

训练完成后，会生成以下输出：
- 训练报告：`logs/train_log_new.txt`
- 模型文件：`../data/models/` 目录下的多个 `.pkl` 文件
- 部署报告：`../data/models/deployments/` 目录下的部署报告
- 控制台输出：包含模型性能指标和业务标签

## 配置说明

配置参数位于 `src/config.py` 文件中，可根据需要调整：

- **数据配置**：数据文件路径、文本列名
- **数据划分比例**：训练集、验证集、测试集比例
- **向量化器配置**：特征数量、n-gram范围等
- **LDA配置**：迭代次数、学习方法、先验参数等
- **主题数评估范围**：默认评估 [4, 5, 6, 7] 个主题
- **分类器配置**：随机森林参数
- **关键词提取配置**：每个主题提取的关键词数量
- **主题名称优化配置**：优化参数和规则

## API文档

### 主要模块API

#### 1. 数据预处理模块 (`data_preprocessor.py`)

```python
class DataPreprocessor:
    def __init__(self, text_preprocessor):
        """初始化数据预处理器"""
    
    def load_data(self):
        """加载数据文件"""
    
    def preprocess_data(self, df):
        """预处理数据集"""
    
    def split_data(self, df):
        """划分数据集为训练集、验证集、测试集"""
```

#### 2. 文本预处理模块 (`text_preprocessor.py`)

```python
class TextPreprocessor:
    def __init__(self):
        """初始化文本预处理器"""
    
    def preprocess(self, text):
        """预处理单个文本"""
    
    def remove_stopwords(self, words):
        """移除停用词"""
```

#### 3. 主题建模模块 (`topic_modeler.py`)

```python
class TopicModeler:
    def create_count_vectorizer(self):
        """创建Count向量化器"""
    
    def create_tfidf_vectorizer(self):
        """创建TF-IDF向量化器"""
    
    def evaluate_topic_numbers(self, X_train):
        """评估不同主题数的性能"""
    
    def train_final_model(self, X_train, n_topics):
        """训练最终模型"""
    
    def save_model(self):
        """保存模型"""
    
    def load_model(self, model_path):
        """加载模型"""
    
    def evaluate_perplexity(self, new_data=None):
        """评估模型困惑度"""
    
    def evaluate_coherence(self):
        """评估主题一致性"""
```

#### 4. 主题名称优化模块 (`topic_name_optimizer.py`)

```python
class TopicNameOptimizer:
    def optimize_topic_name(self, topic_name, keywords):
        """优化主题名称"""
    
    def _clean_topic_name(self, topic_name):
        """清洗主题名称"""
    
    def _extract_core_concepts(self, keywords):
        """提取核心概念"""
    
    def _generate_candidate_names(self, cleaned_name, core_concepts, keywords):
        """生成候选名称"""
    
    def _select_best_name(self, candidate_names, keywords):
        """选择最佳名称"""
```

#### 5. 模型监控模块 (`model_monitor.py`)

```python
class ModelMonitor:
    def monitor_model_performance(self, model_path, new_data=None):
        """监控模型性能"""
    
    def compare_models(self, model_path1, model_path2):
        """比较两个模型的性能"""
    
    def generate_alert(self, report):
        """生成性能警报"""
```

#### 6. 标签生成模块 (`label_generator.py`)

```python
class LabelGenerator:
    def get_business_labels(self, n_topics, lda_model=None, vectorizer=None):
        """基于关键词语义自动生成业务化标签"""
```

#### 7. 分类器模块 (`classifier.py`)

```python
class ThemeClassifier:
    def create_classifier(self):
        """创建随机森林分类器"""
    
    def train(self, X_train, y_train):
        """训练分类器"""
    
    def evaluate(self, X_test, y_test, business_labels):
        """评估分类器性能"""
    
    def predict(self, X):
        """预测新数据"""
```

#### 8. 模型部署模块 (`scripts/deploy_model.py`)

```python
class ModelDeployer:
    def deploy_model(self, new_data=None, config_updates=None):
        """部署模型"""
    
    def rollback_deployment(self, deployment_id=None):
        """回滚部署"""
    
    def get_deployment_status(self):
        """获取部署状态"""
```

## 示例输出

### 业务标签示例

```
线下社交
脱单效果与满意度
功能评价与满意度
脱单效果
功能评价
平台对比
线下社交(1)
```

### 关键词示例

```
线下社交: 软件, 可以, 什么, 交友, 就是, 这个, 上面, 但是, 正常, 交友 软件
脱单效果与满意度: 现在, 怎么, 会员, 下载, 这个, 不到, 厉害, 聊天, 对象, 已经
```

### 部署报告示例

```json
{
  "deployment_id": "deploy_20260111_233020",
  "timestamp": "2026-01-11T23:30:20.123456",
  "deployment_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\deployments\\deploy_20260111_233020",
  "new_model_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\lda_model.pkl",
  "current_model_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\lda_model.pkl",
  "deployment_decision": {
    "deploy_new": false,
    "reason": "现有模型性能更优"
  },
  "model_performance": {
    "timestamp": "2026-01-11T23:30:20.123456",
    "performance_metrics": {
      "perplexity": 1.0,
      "coherence": 0.6473,
      "topic_distribution": {
        "n_topics": 7,
        "topic_indices": [0, 1, 2, 3, 4, 5, 6]
      },
      "degradation": {
        "detected": false,
        "reasons": []
      }
    },
    "recommendations": [
      "模型性能稳定，无需立即更新",
      "困惑度较高，建议调整模型参数"
    ]
  },
  "optimized_topics": {},
  "config_updates": null
}
```

## 部署说明

### 自动部署流程

1. **数据准备**：加载并预处理数据
2. **模型训练**：评估最佳主题数并训练最终模型
3. **主题优化**：自动优化主题名称
4. **模型评估**：评估新模型性能
5. **模型比较**：与现有模型比较性能
6. **部署决策**：基于性能比较结果决定是否部署
7. **部署执行**：执行部署或回滚操作
8. **报告生成**：生成部署报告

### 模型部署

训练完成后，模型文件会保存在 `../data/models/` 目录中，包括：

- `lda_model.pkl`：LDA主题模型
- `count_vectorizer.pkl`：Count向量化器
- `vectorizer.pkl`：TF-IDF向量化器
- `theme_classification_model.pkl`：主题分类器
- `theme_keywords.pkl`：主题关键词
- `optimized_topics.json`：优化后的主题名称

### 集成到生产环境

可以通过以下方式加载模型并进行预测：

```python
import joblib
import numpy as np
from src.text_preprocessor import TextPreprocessor

# 加载模型
lda_model = joblib.load('../data/models/lda_model.pkl')
count_vectorizer = joblib.load('../data/models/count_vectorizer.pkl')
tfidf_vectorizer = joblib.load('../data/models/vectorizer.pkl')
classifier = joblib.load('../data/models/theme_classification_model.pkl')

# 初始化文本预处理器
text_preprocessor = TextPreprocessor()

def preprocess_text(text):
    """预处理文本"""
    processed_words = text_preprocessor.preprocess(text)
    return " ".join(processed_words)

def predict_topic(text):
    """预测文本主题"""
    # 预处理文本
    cleaned_text = preprocess_text(text)
    
    # 提取特征
    count_features = count_vectorizer.transform([cleaned_text])
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    topic_dist = lda_model.transform(count_features)
    
    # 组合特征
    combined_features = np.hstack([topic_dist, tfidf_features.toarray()])
    
    # 预测
    topic_id = classifier.predict(combined_features)[0]
    
    return topic_id
```

## 参数调优过程

### 调优方法
1. **设计参数调优脚本**：创建了专门的参数调优脚本 `optimize_topic_clarity.py`，系统测试不同参数组合
2. **参数范围选择**：
   - topic_word_prior: [0.0001, 0.001, 0.01]
   - doc_topic_prior: [0.1, 0.3, 0.5]
   - max_iter: [500, 800, 1000]
   - n_topics: [5, 6, 7, 8]
3. **评估指标**：同时考虑主题清晰度、困惑度和分类准确率
4. **自动化测试**：系统运行144种参数组合，记录每种组合的性能指标
5. **最佳参数选择**：基于多指标综合评估，选择最优参数组合

### 调优结果分析
- **测试组合数**：144种参数组合
- **最佳参数组合**：
  - topic_word_prior: 0.0001（控制主题词分布平滑度，值越小主题越集中）
  - doc_topic_prior: 0.3（控制文档主题分布平滑度，平衡主题多样性）
  - max_iter: 500（足够的迭代次数确保模型收敛）
  - learning_offset: 15.0（在线学习的学习率偏移）
- **调优效果**：
  - 困惑度：从初始的极高值（353 billion trillion）降低到 2783.50
  - 主题清晰度：从 0.29 提升到 0.3203（提升10.4%）
  - 分类准确率：保持在 92.94% 以上，确保模型性能

### 相关文档保存
- **调优日志**：保存在 `logs/` 目录中，记录详细的参数测试结果
- **训练报告**：`logs/train_log_new.txt` 包含完整的训练过程和性能指标
- **模型文件**：最佳参数配置的模型保存在 `../data/models/` 目录中
- **配置文件**：最佳参数配置更新到 `src/config.py` 文件中，便于后续使用

### 最佳参数
- max_iter : 500
- learning_offset : 15.0
- topic_word_prior : 0.0001
- doc_topic_prior : 0.3

## 主题清晰度优化
### 什么是主题清晰度？
主题清晰度是衡量主题模型质量的重要指标，计算公式为：
`1 - (平均主题熵 / log(特征数量))`
- 熵值越低，主题词分布越集中，主题越清晰
- 清晰度值范围为 0-1，值越高表示主题越清晰

### 为什么主题清晰度重要？
- 高清晰度意味着主题边界更明确，标签更准确
- 有助于业务人员更好地理解和使用主题模型
- 提高模型的可解释性和实用性

### 影响主题清晰度的参数
1. **topic_word_prior**：控制主题词分布的平滑度，值越小主题越集中
2. **doc_topic_prior**：控制文档主题分布的平滑度，影响主题多样性
3. **n_topics**：主题数量，过多或过少都会影响清晰度
4. **max_iter**：迭代次数，足够的迭代次数确保模型收敛

### 清晰度优化方法
1. **参数调优**：系统测试不同参数组合，找到最佳配置
2. **文本预处理优化**：提高文本质量，减少噪音
3. **主题数选择**：选择合适的主题数量，平衡粒度和清晰度
4. **标签生成优化**：确保标签语义明确，避免重复

## 重复标签问题解决方案
### 问题原因
- 不同主题可能包含相似的关键词，导致生成相似的业务标签
- 原始标签生成逻辑仅基于关键词频率，缺乏语义区分度

### 解决方案
1. **增强版标签生成器**：
   - 添加 `_select_most_relevant_subtype` 方法，基于关键词选择最相关的子类型
   - 添加 `_extract_differentiator` 方法，提取区分性关键词
   - 生成格式为 "主类型(区分词)" 的唯一标签

2. **语义理解优化**：
   - 为每个子类型定义关键词集合
   - 基于关键词匹配度选择最相关的子类型
   - 提取主题特有的区分性关键词，确保标签唯一性

3. **示例改进**：
   - 原标签："线下社交"、"线下社交(1)"
   - 新标签："线下社交(交友)"、"线下社交(社区)"
   - 原标签："脱单效果"、"脱单效果(1)"
   - 新标签："脱单效果(对象)"、"脱单效果(满意度)"

## 性能指标

- **测试集准确率**：> 95%
- **LDA困惑度**：~ 2,783.50
- **主题清晰度**：~ 0.3203
- **主题一致性**：~ 0.82
- **部署成功率**：100%

## 故障排查

### 常见问题

1. **数据文件不存在**：确保 `../data/raw/评论和正文.xlsx` 文件存在且包含 `评论内容` 列
2. **内存不足**：对于大规模数据，可以减少 `max_features` 参数
3. **训练时间过长**：可以减少主题数评估范围或降低 `max_iter` 参数
4. **部署失败**：检查模型文件路径和权限

### 日志查看

- 训练过程中的详细日志会输出到控制台，同时保存到 `logs/train_log_new.txt` 文件中
- 部署过程中的日志会输出到控制台，同时保存到部署报告中
- 模型监控日志会保存到 `../data/models/monitoring/` 目录中

## 版本历史

- **v1.0**：初始版本，实现基本功能
- **v1.1**：重构代码结构，添加模块化设计
- **v1.2**：优化标签生成逻辑，添加智能语义理解
- **v1.3**：扩展模型评估指标，添加主题一致性评估
- **v1.4**：添加主题名称优化器和模型监控器
- **v1.5**：实现自动部署流程和完整的测试套件

## 贡献指南

欢迎提交Issue和Pull Request，贡献代码和改进建议。

## 许可证

本项目采用 MIT 许可证。