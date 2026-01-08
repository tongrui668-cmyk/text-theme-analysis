import pandas as pd
import numpy as np
import re
import jieba
import joblib
import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📋 1. 加载数据文件
file_path = '../data/raw/评论和正文.xlsx'
print(f"加载数据文件: {file_path}")
df = pd.read_excel(file_path)

# 检查是否包含 '评论内容' 列
if '评论内容' not in df.columns:
    raise ValueError("数据文件中缺少 '评论内容' 列，请检查文件。")

print(f"数据加载完成，共 {len(df)} 条记录")

# 📋 2. 文本预处理函数（改进版）
def clean_text(text):
    if pd.isnull(text):
        return ""
    
    # 基本清洗，去除特殊字符但保留中文、英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    # 使用jieba分词，添加自定义词典
    jieba.load_userdict('../training/custom_dict.txt')
    words = jieba.lcut(text)
    
    # 过滤掉单字符和停用词
    stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    filtered_words = [word.strip() for word in words if len(word.strip()) > 1 and word not in stop_words]
    
    return " ".join(filtered_words)

# 应用文本预处理
print("开始文本预处理...")
df['cleaned_text'] = df['评论内容'].apply(clean_text)

# 过滤掉空的文本
df = df[df['cleaned_text'].str.len() > 0]
print(f"预处理完成，过滤后剩余 {len(df)} 条有效记录")

# 📋 3. 向量化文本数据（改进参数）
print("开始向量化文本数据...")
# 使用更合理的参数
vectorizer = CountVectorizer(
    max_features=8000,      # 增加特征数量
    min_df=2,               # 至少在2个文档中出现
    max_df=0.9,             # 最多在90%的文档中出现
    ngram_range=(1, 2)      # 包含1-gram和2-gram
)
X = vectorizer.fit_transform(df['cleaned_text'])
print(f"向量化完成，特征维度: {X.shape[1]}")

# 📋 4. 训练LDA模型（改进参数）
print("开始训练LDA模型...")
n_topics = 5  # 保持5个主题，与现有系统一致

# 使用改进的参数
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=50,           # 增加迭代次数
    learning_method='online',  # 使用在线学习方法
    learning_offset=10.0,     # 学习率偏移
    doc_topic_prior=0.1,       # 文档主题先验
    topic_word_prior=0.01      # 主题词先验
)
lda.fit(X)
print("LDA模型训练完成")

# 📋 5. 输出每个主题的关键词
def get_lda_keywords(model, vectorizer, n_top_words=15):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        keywords[f'主题 {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return keywords

# 提取关键词
keywords = get_lda_keywords(lda, vectorizer)
print("\n自动生成的主题关键词：")
for theme, words in keywords.items():
    print(f"{theme}: {', '.join(words[:10])}")

# 📋 6. 为每条文本分配主题
print("为文本分配主题...")
doc_topic_dist = lda.transform(X)
df['dominant_topic'] = doc_topic_dist.argmax(axis=1)

# 📋 7. 准备TF-IDF特征
print("创建TF-IDF向量化器...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=8000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2)
)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])
print(f"TF-IDF向量化完成，特征维度: {X_tfidf.shape[1]}")

# 📋 8. 合并特征
print("合并特征...")
X_combined = np.hstack([doc_topic_dist, X_tfidf.toarray()])
print(f"合并后特征维度: {X_combined.shape[1]}")

# 📋 9. 训练主题分类器（改进参数）
print("开始训练主题分类器...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, df['dominant_topic'], test_size=0.2, random_state=42
)

# 使用改进的RandomForest参数
theme_classifier = RandomForestClassifier(
    n_estimators=100,        # 增加树的数量
    max_depth=50,            # 增加树的深度
    min_samples_split=5,     # 最小分裂样本数
    min_samples_leaf=2,      # 最小叶节点样本数
    random_state=42,
    n_jobs=-1                # 使用所有核心
)
theme_classifier.fit(X_train, y_train)

# 📋 10. 评估模型
print("评估模型性能...")
y_pred = theme_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n主题分类器准确率: {accuracy:.4f}")

# 打印详细评估报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=[f'主题 {i+1}' for i in range(n_topics)]))

# 计算困惑度
perplexity = lda.perplexity(X)
print(f"\nLDA模型困惑度: {perplexity:.2f}")

# 📋 11. 保存模型和向量化器
model_dir = '../data/models'
os.makedirs(model_dir, exist_ok=True)

print(f"\n保存模型到 {model_dir}...")
joblib.dump(lda, os.path.join(model_dir, 'lda_model.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'count_vectorizer.pkl'))
joblib.dump(tfidf_vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump(theme_classifier, os.path.join(model_dir, 'theme_classification_model.pkl'))
joblib.dump(keywords, os.path.join(model_dir, 'theme_keywords.pkl'))

print("\n所有模型已保存完成！")
print(f"- LDA模型: lda_model.pkl")
print(f"- Count向量化器: count_vectorizer.pkl")
print(f"- TF-IDF向量化器: vectorizer.pkl")
print(f"- 主题分类器: theme_classification_model.pkl")
print(f"- 主题关键词: theme_keywords.pkl")

# 📋 12. 生成训练报告
print("\n训练完成，生成训练报告...")
with open('train_log_new.txt', 'w', encoding='utf-8') as f:
    f.write("=== 模型训练报告 ===\n")
    f.write(f"训练时间: {pd.Timestamp.now()}\n")
    f.write(f"训练数据量: {len(df)}\n")
    f.write(f"特征维度: {X.shape[1]}\n")
    f.write(f"分类器准确率: {accuracy:.4f}\n")
    f.write(f"LDA困惑度: {perplexity:.2f}\n\n")
    f.write("主题关键词:\n")
    for theme, words in keywords.items():
        f.write(f"{theme}: {', '.join(words[:10])}\n")

print("训练报告已生成: train_log_new.txt")