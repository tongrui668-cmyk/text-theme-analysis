# API 文档

本文档详细介绍了主题模型训练系统的各个模块及其API接口。

## 1. 配置管理模块 (`src/config.py`)

### Config 类

**功能**：管理所有训练相关的配置参数

**属性**：
- `DATA_FILE`：数据文件路径，默认 `'../data/raw/评论和正文.xlsx'`
- `TEXT_COLUMN`：文本列名，默认 `'评论内容'`
- `TRAIN_SIZE`：训练集比例，默认 `0.6`
- `VALIDATION_SIZE`：验证集比例，默认 `0.2`
- `TEST_SIZE`：测试集比例，默认 `0.2`
- `COUNT_VECTORIZER`：Count向量化器配置
- `TFIDF_VECTORIZER`：TF-IDF向量化器配置
- `LDA`：LDA模型配置
- `TOPIC_NUMBERS`：主题数评估范围，默认 `[4, 5, 6, 7]`
- `RANDOM_FOREST`：随机森林分类器配置
- `KEYWORDS`：关键词提取配置
- `MODEL_DIR`：模型保存目录，默认 `'../data/models'`
- `LOG_DIR`：日志保存目录，默认 `'logs'`
- `RANDOM_SEED`：随机种子，默认 `42`

**使用示例**：
```python
from src.config import config
print(config.DATA_FILE)  # 输出数据文件路径
```

## 2. 数据预处理模块 (`src/data_preprocessor.py`)

### DataPreprocessor 类

**功能**：处理数据加载、文本预处理和数据划分

**方法**：
- `__init__(self, text_preprocessor)`：初始化数据预处理器
  - `text_preprocessor`：文本预处理器实例
- `load_data(self)`：加载数据文件
  - **返回**：加载的数据框
- `preprocess_text(self, text)`：预处理单条文本
  - `text`：原始文本
  - **返回**：预处理后的文本
- `preprocess_data(self, df)`：预处理数据集
  - `df`：原始数据框
  - **返回**：预处理后的数据框
- `split_data(self, df)`：划分数据集
  - `df`：预处理后的数据框
  - **返回**：(训练集, 验证集, 测试集) 元组

**使用示例**：
```python
from src.data_preprocessor import DataPreprocessor
from src.text_preprocessor import TextPreprocessor

text_preprocessor = TextPreprocessor()
data_preprocessor = DataPreprocessor(text_preprocessor)
df = data_preprocessor.load_data()
df = data_preprocessor.preprocess_data(df)
df_train, df_val, df_test = data_preprocessor.split_data(df)
```

## 3. 主题建模模块 (`src/topic_modeler.py`)

### TopicModeler 类

**功能**：处理文本向量化和LDA主题建模

**方法**：
- `create_count_vectorizer(self)`：创建Count向量化器
  - **返回**：配置好的CountVectorizer实例
- `create_tfidf_vectorizer(self)`：创建TF-IDF向量化器
  - **返回**：配置好的TfidfVectorizer实例
- `evaluate_topic_numbers(self, X_train)`：评估不同主题数的性能
  - `X_train`：训练集特征矩阵
  - **返回**：(最佳主题数, 最低困惑度, 评估结果) 元组
- `train_lda(self, X_train, n_topics)`：训练最终LDA模型
  - `X_train`：训练集特征矩阵
  - `n_topics`：主题数量
  - **返回**：训练好的LDA模型
- `get_topic_distribution(self, X)`：获取文档-主题分布
  - `X`：特征矩阵
  - **返回**：文档-主题分布矩阵

**使用示例**：
```python
from src.topic_modeler import TopicModeler

topic_modeler = TopicModeler()
vectorizer = topic_modeler.create_count_vectorizer()
X_train_count = vectorizer.fit_transform(df_train['cleaned_text'])
n_topics, best_perplexity, evaluation_results = topic_modeler.evaluate_topic_numbers(X_train_count)
lda = topic_modeler.train_lda(X_train_count, n_topics)
```

## 4. 标签生成模块 (`src/label_generator.py`)

### LabelGenerator 类

**功能**：基于关键词语义自动生成业务化标签

**方法**：
- `__init__(self)`：初始化标签生成器
- `get_business_labels(self, n_topics, lda_model=None, vectorizer=None)`：生成业务标签
  - `n_topics`：主题数量
  - `lda_model`：LDA模型（可选）
  - `vectorizer`：向量化器（可选）
  - **返回**：业务标签字典，格式 `{主题编号: 业务标签}`
- `_generate_label_from_keywords(self, top_words, topic_idx)`：从关键词生成标签
  - `top_words`：主题的关键词列表
  - `topic_idx`：主题编号
  - **返回**：生成的业务标签
- `_extract_specific_dimension(self, top_words)`：从关键词中提取具体维度
  - `top_words`：关键词列表
  - **返回**：具体维度描述
- `_generate_descriptive_label(self, top_words, topic_idx)`：生成描述性标签
  - `top_words`：关键词列表
  - `topic_idx`：主题编号
  - **返回**：描述性标签
- `_ensure_label_uniqueness(self, label)`：确保标签唯一性
  - `label`：原始标签
  - **返回**：唯一标签

**使用示例**：
```python
from src.label_generator import LabelGenerator

label_generator = LabelGenerator()
business_labels = label_generator.get_business_labels(n_topics, lda, vectorizer)
print(business_labels)  # 输出业务标签字典
```

## 5. 关键词提取模块 (`src/keyword_extractor.py`)

### KeywordExtractor 类

**功能**：从LDA模型和TF-IDF中提取关键词

**方法**：
- `__init__(self)`：初始化关键词提取器
- `get_lda_keywords(self, model, vectorizer, business_labels, n_top_words=None)`：从LDA模型提取关键词
  - `model`：LDA模型
  - `vectorizer`：向量化器
  - `business_labels`：业务标签字典
  - `n_top_words`：提取的关键词数量（可选）
  - **返回**：每个主题的关键词字典
- `get_tfidf_keywords_by_topic(self, df, tfidf_vectorizer, business_labels, n_top_words=None)`：基于TF-IDF提取关键词
  - `df`：包含文本和主题的数据框
  - `tfidf_vectorizer`：TF-IDF向量化器
  - `business_labels`：业务标签字典
  - `n_top_words`：提取的关键词数量（可选）
  - **返回**：每个主题的TF-IDF关键词字典
- `get_combined_keywords(self, lda_keywords, tfidf_keywords, n_top_words=None)`：结合LDA和TF-IDF关键词
  - `lda_keywords`：LDA主题关键词
  - `tfidf_keywords`：TF-IDF主题关键词
  - `n_top_words`：每个主题保留的关键词数量（可选）
  - **返回**：每个主题的组合关键词字典

**使用示例**：
```python
from src.keyword_extractor import KeywordExtractor

keyword_extractor = KeywordExtractor()
lda_keywords = keyword_extractor.get_lda_keywords(lda, vectorizer, business_labels)
tfidf_keywords = keyword_extractor.get_tfidf_keywords_by_topic(df_train, tfidf_vectorizer, business_labels)
combined_keywords = keyword_extractor.get_combined_keywords(lda_keywords, tfidf_keywords)
```

## 6. 分类器模块 (`src/classifier.py`)

### ThemeClassifier 类

**功能**：训练和评估主题分类器

**方法**：
- `__init__(self)`：初始化分类器
- `create_classifier(self)`：创建随机森林分类器
  - **返回**：配置好的RandomForestClassifier实例
- `train(self, X_train, y_train)`：训练分类器
  - `X_train`：训练特征
  - `y_train`：训练标签
  - **返回**：训练好的分类器
- `evaluate(self, X_test, y_test, business_labels)`：评估分类器性能
  - `X_test`：测试特征
  - `y_test`：测试标签
  - `business_labels`：业务标签字典
  - **返回**：评估结果字典，包含准确率、混淆矩阵等
- `predict(self, X)`：预测新数据
  - `X`：新数据特征
  - **返回**：预测结果

**使用示例**：
```python
from src.classifier import ThemeClassifier

classifier = ThemeClassifier()
classifier.create_classifier()
classifier.train(X_combined_train, df_train['dominant_topic'])
evaluation_results = classifier.evaluate(X_combined_test, df_test['dominant_topic'], business_labels)
```

## 7. 模型保存模块 (`src/model_saver.py`)

### ModelSaver 类

**功能**：保存训练好的模型和向量化器

**方法**：
- `__init__(self)`：初始化模型保存器
- `save_models(self, models)`：保存模型和向量化器
  - `models`：模型字典，包含所有需要保存的模型
  - **返回**：保存的模型路径字典

**使用示例**：
```python
from src.model_saver import ModelSaver

model_saver = ModelSaver()
models_to_save = {
    'lda': lda,
    'count_vectorizer': vectorizer,
    'tfidf_vectorizer': tfidf_vectorizer,
    'classifier': classifier.classifier,
    'keywords': combined_keywords
}
model_paths = model_saver.save_models(models_to_save)
```

## 8. 报告生成模块 (`src/report_generator.py`)

### ReportGenerator 类

**功能**：生成详细的训练报告

**方法**：
- `__init__(self)`：初始化报告生成器
- `generate_report(self, training_results)`：生成训练报告
  - `training_results`：训练结果字典
  - **返回**：报告文件路径

**使用示例**：
```python
from src.report_generator import ReportGenerator

report_generator = ReportGenerator()
report_path = report_generator.generate_report(training_results)
print(f"训练报告已生成: {report_path}")
```

## 9. 文本预处理模块 (`src/text_preprocessor.py`)

### TextPreprocessor 类

**功能**：处理文本清洗、分词、去停用词等操作

**方法**：
- `__init__(self)`：初始化文本预处理器
- `clean_text(self, text)`：清洗文本
  - `text`：原始文本
  - **返回**：清洗后的文本
- `tokenize(self, text)`：分词
  - `text`：清洗后的文本
  - **返回**：分词结果列表
- `remove_duplicate_phrases(self, words)`：移除重复短语
  - `words`：分词结果
  - **返回**：去重后的词列表
- `_is_repetitive_interjection(self, word)`：判断是否为重复的语气词
  - `word`：词语
  - **返回**：是否为重复语气词
- `remove_stopwords(self, words)`：移除停用词和无意义的语气词
  - `words`：分词结果
  - **返回**：去停用词后的词列表
- `preprocess(self, text)`：完整的文本预处理流程
  - `text`：原始文本
  - **返回**：预处理后的词列表

**使用示例**：
```python
from src.text_preprocessor import TextPreprocessor

text_preprocessor = TextPreprocessor()
processed_words = text_preprocessor.preprocess("这是一段测试文本，哈哈哈哈")
print(processed_words)  # 输出预处理后的词列表
```

## 10. 主训练脚本 (`train.py`)

**功能**：整合所有模块，执行完整的训练流程

**执行流程**：
1. 初始化所有组件
2. 加载和预处理数据
3. 向量化文本数据
4. 评估最佳主题数
5. 训练LDA模型
6. 生成业务标签
7. 提取关键词
8. 训练分类器
9. 评估模型性能
10. 生成训练报告
11. 保存模型

**使用方式**：
```bash
python train.py
```

## 输入输出示例

### 输入输出示例

**输入**：
```python
# 运行训练脚本
python train.py
```

**输出**：
```
初始化训练组件...
加载数据文件: ../data/raw/评论和正文.xlsx
数据加载完成，共 6000 条记录
开始文本预处理...
预处理完成，过滤后剩余 5526 条有效记录
开始数据划分...
数据划分完成:
- 训练集: 3315 条记录
- 验证集: 1105 条记录
- 测试集: 1106 条记录
开始向量化文本数据（仅使用训练集）...
创建Count向量化器...
向量化完成，特征维度: 6141
开始自动评估最佳主题数...
评估主题数: 4...
  困惑度: 5305.04, 主题清晰度: 0.2051, 主题一致性: 0.9048
评估主题数: 5...
  困惑度: 4945.02, 主题清晰度: 0.2305, 主题一致性: 0.9238
评估主题数: 6...
  困惑度: 4702.63, 主题清晰度: 0.2433, 主题一致性: 0.8194
评估主题数: 7...
  困惑度: 4350.51, 主题清晰度: 0.2592, 主题一致性: 0.8452

最佳主题数: 7
最低困惑度: 4350.51

开始训练最终LDA模型（主题数=7）...
LDA模型训练完成
基于关键词自动生成业务标签...
业务标签生成完成
为训练集分配主题...
创建TF-IDF向量化器（仅使用训练集）...
创建TF-IDF向量化器...
TF-IDF向量化完成，特征维度: 6141

自动生成的主题关键词：
线下社交: 软件, 可以, 什么, 交友, 就是, 这个, 上面, 但是, 正常, 交友 软件
脱单效果与满意度: 现在, 怎么, 会员, 下载, 这个, 不到, 厉害, 聊天, 对象, 已经
功能评价与满意度: 软件, 感觉, 知道, 你们, 这个, 那个, app, tinder, 什么, 这个 app        
脱单效果: 不如, 这些, 好玩, 剩下, 看到, 剩下 不如, 这些 剩下, 恋爱, 本人, 不用
功能评价: 可以, 社交, app, 一起, 分享, 遇到, 有没有, 就是, 直接, 软件
平台对比: soul, 感觉, 不是, 认识, 有点, 真的, 因为, 二狗, 不错, 评论
线下社交(1): 真的, 什么, 美团, app, 可以, 线下, 一下, 有人, 两个, 找对象

# ... 中间输出省略 ...

测试集准确率: 0.9322

分类报告（测试集）:
              precision    recall  f1-score   support

        线下社交       0.86      0.99      0.92       266
    脱单效果与满意度       0.96      0.92      0.94       211
    功能评价与满意度       0.97      0.88      0.93        86
        脱单效果       0.96      0.94      0.95       107
        功能评价       0.97      0.89      0.93       176
        平台对比       0.94      0.93      0.93       137
     线下社交(1)       0.96      0.91      0.93       123

    accuracy                           0.93      1106
   macro avg       0.95      0.92      0.93      1106
weighted avg       0.94      0.93      0.93      1106

# ... 后续输出省略 ...

==================================================
训练完成！
==================================================
最佳主题数: 7
测试集准确率: 0.9322
LDA困惑度: 4350.51
训练报告: logs\train_log_new.txt
模型保存路径: ../data/models

主题业务标签：
0: 线下社交
1: 脱单效果与满意度
2: 功能评价与满意度
3: 脱单效果
4: 功能评价
5: 平台对比
6: 线下社交(1)
```