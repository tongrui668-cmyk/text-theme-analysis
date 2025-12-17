import pandas as pd
import re
import jieba
import jieba.posseg as pseg
from jieba.analyse import extract_tags

# 📌 第一步：加载数据和初步清洗
file_path = '../data/raw/评论和正文.xlsx'  # 数据文件路径
df = pd.read_excel(file_path)

# 初步清洗：去除非中文字符
def clean_text(text):
    return re.sub(r'[^\u4e00-\u9fa5]', '', text)

df['评论内容_清洗后'] = df['评论内容'].apply(lambda x: clean_text(str(x)) if pd.notnull(x) else '')

# 查看清洗后的数据
print("初步清洗后的数据：")
print(df[['评论内容', '评论内容_清洗后']].head())

# 📌 第二步：加载自定义词典
jieba.load_userdict('custom_dict.txt')

# 📌 第二步：中文分词
def tokenize(text):
    return " ".join(jieba.cut(text))

df['评论内容_分词'] = df['评论内容_清洗后'].apply(tokenize)

# 查看分词结果
print("\n分词后的数据：")
print(df[['评论内容_清洗后', '评论内容_分词']].head())

# 📌 第三步：停用词过滤
# 加载停用词表
stopwords_path = '../data/raw/chinese_stopwords.txt'  # 停用词文件路径
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

# 过滤停用词
def remove_stopwords(text):
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])

df['评论内容_去停用词'] = df['评论内容_分词'].apply(remove_stopwords)

# 查看去停用词结果
print("\n去停用词后的数据：")
print(df[['评论内容_分词', '评论内容_去停用词']].head())

# 📌 第四步：词性标注
def pos_tagging(text):
    words = pseg.cut(text)
    return [(word, flag) for word, flag in words]

df['评论内容_词性标注'] = df['评论内容_清洗后'].apply(pos_tagging)

# 查看词性标注结果
print("\n词性标注后的数据：")
print(df[['评论内容_清洗后', '评论内容_词性标注']].head())

# 📌 第五步：关键词提取
def extract_keywords(text, topK=5):
    return extract_tags(text, topK=topK)

df['评论内容_关键词'] = df['评论内容_清洗后'].apply(lambda x: extract_keywords(x, topK=5))

# 查看关键词提取结果
print("\n关键词提取结果：")
print(df[['评论内容_清洗后', '评论内容_关键词']].head())

# 📌 保存预处理后的数据到新 Excel 文件
output_path = '../data/raw/预处理后的评论数据.xlsx'
df.to_excel(output_path, index=False)
print(f"\n预处理后的数据已保存至：{output_path}")
