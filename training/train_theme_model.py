import pandas as pd
import numpy as np
import re
import thulac  # 替换 jieba 为 thulac
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 初始化分词器
thu = thulac.thulac(seg_only=True)

# 📋 1. 加载数据文件
file_path = '../data/raw/评论和正文.xlsx'
if not os.path.exists(file_path):
    file_path = 'data/raw/评论和正文.xlsx'
df = pd.read_excel(file_path)

# 检查数据列是否存在
if '评论内容' not in df.columns:
    raise ValueError("数据文件中缺少 '评论内容' 列，请检查文件。")

# 📋 2. 文本预处理和分词
def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 基本清洗，去除特殊字符但保留中文、英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # THULAC分词
    words = thu.cut(text, text=True)
    # 过滤掉单字符和停用词
    filtered_words = [word.strip() for word in words.split() if len(word.strip()) > 1]
    return ' '.join(filtered_words)

df['cleaned_text'] = df['评论内容'].apply(clean_text)
print(f"预处理后文本样本: {df['cleaned_text'].iloc[0] if len(df) > 0 else 'None'}")
print(f"非空文本数量: {df['cleaned_text'].str.len().gt(0).sum()}")

# 过滤掉空的文本
df = df[df['cleaned_text'].str.len() > 0]
print(f"过滤后数据量: {len(df)}")

# 📋 3. 动态调整主题数量
def find_optimal_topics(X, start=2, end=5):
    print("开始寻找最佳主题数...")
    perplexities = []
    for n_topics in range(start, end + 1):
        print(f"测试主题数: {n_topics}")
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
        lda.fit(X)
        perplexity = lda.perplexity(X)
        perplexities.append((n_topics, perplexity))
        print(f"主题数 {n_topics}: 困惑度 = {perplexity:.2f}")
    optimal_topics = min(perplexities, key=lambda x: x[1])[0]
    return optimal_topics

# 向量化数据
print("开始向量化数据...")
vectorizer = CountVectorizer(max_features=5000, min_df=1, max_df=0.95)
X = vectorizer.fit_transform(df['cleaned_text'])
print(f"Count向量化完成，特征维度: {X.shape[1]}")

# 动态确定最佳主题数
print("使用固定主题数进行训练...")
n_topics = 5  # 直接使用固定主题数
print(f"确定主题数: {n_topics}")
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(X)
print("LDA模型训练完成")

# 📋 4. 提取主题关键词
def get_lda_keywords(model, vectorizer, n_top_words=10):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        keywords[f'主题 {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return keywords

theme_keywords = get_lda_keywords(lda, vectorizer)

# 打印主题关键词
print("\n自动生成的主题关键词：")
for theme, words in theme_keywords.items():
    print(f"{theme}: {', '.join(words)}")

# 📋 5. 训练主题分类器
print("开始训练主题分类器...")
# 为每条文本分配主要主题
doc_topic_dist = lda.transform(X)
df['dominant_topic'] = doc_topic_dist.argmax(axis=1)

# 准备训练数据：合并LDA特征和TF-IDF特征
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建TF-IDF向量化器
print("创建TF-IDF向量化器...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.95)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
print(f"TF-IDF向量化完成，特征维度: {X_tfidf.shape[1]}")

# 合并特征
print("合并特征...")
X_combined = np.hstack([doc_topic_dist, X_tfidf.toarray()])
print(f"合并后特征维度: {X_combined.shape[1]}")

# 训练主题分类器
print("训练RandomForest分类器...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, df['dominant_topic'], test_size=0.2, random_state=42
)

theme_classifier = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
theme_classifier.fit(X_train, y_train)

# 评估模型
y_pred = theme_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n主题分类器准确率: {accuracy:.4f}")

# 📋 6. 保存所有模型和向量化器
models_dir = '../data/models'
os.makedirs(models_dir, exist_ok=True)
joblib.dump(lda, os.path.join(models_dir, 'lda_model.pkl'))
joblib.dump(vectorizer, os.path.join(models_dir, 'count_vectorizer.pkl'))  # CountVectorizer
joblib.dump(tfidf_vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))  # TF-IDF Vectorizer
joblib.dump(theme_classifier, os.path.join(models_dir, 'theme_classification_model.pkl'))
joblib.dump(theme_keywords, os.path.join(models_dir, 'theme_keywords.pkl'))

print(f"\n所有模型已保存到 {models_dir}")
print(f"- LDA模型: lda_model.pkl")
print(f"- Count向量化器: count_vectorizer.pkl") 
print(f"- TF-IDF向量化器: vectorizer.pkl")
print(f"- 主题分类器: theme_classification_model.pkl")
print(f"- 主题关键词: theme_keywords.pkl")
