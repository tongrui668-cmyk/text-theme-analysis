import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import joblib
import os

# 📋 1. 加载数据文件
file_path = '../data/raw/评论和正文.xlsx'  # 数据文件路径
df = pd.read_excel(file_path)

# 检查是否包含 '评论内容' 列
if '评论内容' not in df.columns:
    raise ValueError("数据文件中缺少 '评论内容' 列，请检查文件。")

# 📋 2. 文本预处理函数
def clean_text(text):
    if pd.isnull(text):
        return ""
    words = jieba.lcut(text)
    return " ".join(words)

# 应用文本预处理
df['cleaned_text'] = df['评论内容'].apply(clean_text)

# 📋 3. 向量化文本数据
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['cleaned_text'])

# 📋 4. 训练 LDA 模型
n_topics = 5  # 可以根据需要调整主题数量
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# 📋 5. 输出每个主题的关键词
def get_lda_keywords(model, vectorizer, n_top_words=10):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        keywords[f'主题 {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return keywords

# 提取关键词
keywords = get_lda_keywords(lda, vectorizer)
print("\n自动生成的主题关键词：")
for theme, words in keywords.items():
    print(f"{theme}: {', '.join(words)}")

# 📋 6. 保存模型和向量化器
model_dir = '../data/models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(lda, os.path.join(model_dir, 'lda_model.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
print("\nLDA 模型和向量化器已保存到 data/models/ 目录！")
