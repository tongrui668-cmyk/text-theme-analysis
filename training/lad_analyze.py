# 导入所需库
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 📌 1. 加载预处理后的数据
file_path = '../data/raw/预处理后的评论数据.xlsx'
df = pd.read_excel(file_path)

# 去除空值评论
df = df.dropna(subset=['评论内容_去停用词'])
texts = df['评论内容_去停用词'].tolist()

# 📌 2. 将文本转换为词频矩阵（CountVectorizer）
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=None)
X = vectorizer.fit_transform(texts)

# 📌 3. 构建 LDA 模型
n_topics = 5  # 设置主题数量
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# 📌 4. 打印每个主题的关键词
def print_top_words(model, feature_names, n_top_words=10):
    topic_labels = []
    for topic_idx, topic in enumerate(model.components_):
        keywords = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(f"\n主题 {topic_idx + 1} 的关键词：{keywords}")
        topic_labels.append(f"主题 {topic_idx + 1}: {keywords.split()[0]}...")
    return topic_labels

topic_labels = print_top_words(lda, vectorizer.get_feature_names_out())

# 📌 5. 绘制主题分布饼图
topic_counts = lda.transform(X).argmax(axis=1)
topic_counts_df = pd.DataFrame({'主题': topic_counts + 1})
topic_distribution = topic_counts_df['主题'].value_counts().sort_index()

plt.figure(figsize=(8, 8))
plt.pie(topic_distribution.values, labels=topic_labels, autopct='%1.1f%%', startangle=140)
plt.title('用户评论主题分布')
plt.savefig('../static/reports/主题分布饼图.png')  # 保存饼图到reports目录
plt.show()

# 📌 7. 提取每个主题的典型评论
df['主题'] = topic_counts + 1  # 将主题分配结果添加到数据框中

print("\n每个主题的典型评论：")
for topic in range(1, n_topics + 1):
    print(f"\n{topic_labels[topic - 1]} 的典型评论：")
    sample_comments = df[df['主题'] == topic]['评论内容'].head(5)
    for i, comment in enumerate(sample_comments):
        print(f"{i + 1}. {comment}")

# 📌 8. 添加主题标签并保存结果到 Excel 文件
df['主题标签'] = df['主题'].apply(lambda x: topic_labels[x - 1])
output_path = '../data/raw/LDA主题分析结果_优化.xlsx'
df.to_excel(output_path, index=False)

print(f"\n主题分配结果已保存为 '{output_path}'")
