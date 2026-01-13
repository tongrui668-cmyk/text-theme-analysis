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
import sys
sys.path.append('../')
from src.text_preprocessor import TextPreprocessor

# 📋 1. 加载数据文件
file_path = '../data/raw/评论和正文.xlsx'
print(f"加载数据文件: {file_path}")
df = pd.read_excel(file_path)

# 检查是否包含 '评论内容' 列
if '评论内容' not in df.columns:
    raise ValueError("数据文件中缺少 '评论内容' 列，请检查文件。")

print(f"数据加载完成，共 {len(df)} 条记录")

# 📋 2. 文本预处理（使用与用户相同的TextPreprocessor）
print("开始文本预处理...")
preprocessor = TextPreprocessor()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    # 使用与用户相同的预处理流程
    processed_words = preprocessor.preprocess(text)
    return " ".join(processed_words)

# 应用文本预处理
df['cleaned_text'] = df['评论内容'].apply(preprocess_text)

# 过滤掉空的文本
df = df[df['cleaned_text'].str.len() > 0]
print(f"预处理完成，过滤后剩余 {len(df)} 条有效记录")

# 📋 3. 数据划分 - 首先进行数据划分，确保测试集完全隔离
print("开始数据划分...")
# 第一步：将数据分为训练集（80%）和测试集（20%），测试集完全隔离
# 暂时不使用stratify，因为dominant_topic还未生成
df_train_val, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)

# 第二步：在训练验证集中再分为训练集（75%）和验证集（25%）
df_train, df_val = train_test_split(
    df_train_val, test_size=0.25, random_state=42
)

print(f"数据划分完成:")
print(f"- 训练集: {len(df_train)} 条记录")
print(f"- 验证集: {len(df_val)} 条记录")
print(f"- 测试集: {len(df_test)} 条记录")

# 📋 4. 向量化文本数据（仅使用训练集）
print("开始向量化文本数据（仅使用训练集）...")
# 使用更合理的参数
vectorizer = CountVectorizer(
    max_features=8000,      # 增加特征数量
    min_df=2,               # 至少在2个文档中出现
    max_df=0.9,             # 最多在90%的文档中出现
    ngram_range=(1, 2)      # 包含1-gram和2-gram
)
X_train_count = vectorizer.fit_transform(df_train['cleaned_text'])
print(f"向量化完成，特征维度: {X_train_count.shape[1]}")

# 自动生成主题业务化标签
def get_business_labels(n_topics, lda_model=None, vectorizer=None):
    """
    基于关键词语义自动生成业务化标签
    
    Args:
        n_topics: 主题数量
        lda_model: LDA模型（用于提取关键词）
        vectorizer: 向量化器（用于获取特征名称）
        
    Returns:
        业务标签字典，格式：{主题编号: 业务标签}
    """
    business_labels = {}
    
    # 如果提供了模型和向量化器，基于关键词自动生成标签
    if lda_model and vectorizer:
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx in range(n_topics):
            # 获取主题的前10个关键词
            topic = lda_model.components_[topic_idx]
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            
            # 基于关键词语义生成标签
            # 1. 提取核心概念
            core_concepts = []
            
            # 社交相关关键词
            social_words = ['社交', '线下', '圈子', '社区', '互动', '交流']
            if any(word in top_words for word in social_words):
                core_concepts.append('社交体验')
            
            # 匹配相关关键词
            match_words = ['匹配', '推荐', '结果', '质量', '精准']
            if any(word in top_words for word in match_words):
                core_concepts.append('匹配效果')
            
            # 脱单相关关键词
            dating_words = ['脱单', '恋爱', '男朋友', '女朋友', '结婚']
            if any(word in top_words for word in dating_words):
                core_concepts.append('脱单效果')
            
            # 用户相关关键词
            user_words = ['用户', '质量', '正常人', '真实', '靠谱']
            if any(word in top_words for word in user_words):
                core_concepts.append('用户质量')
            
            # 功能相关关键词
            feature_words = ['功能', 'app', 'AI', '探探', '抖音']
            if any(word in top_words for word in feature_words):
                core_concepts.append('功能评价')
            
            # 体验相关关键词
            experience_words = ['体验', '好用', '满意', '卸载', '失望']
            if any(word in top_words for word in experience_words):
                core_concepts.append('使用体验')
            
            # 2. 生成标签
            if core_concepts:
                # 组合核心概念
                label = '与'.join(core_concepts[:2])  # 最多取2个核心概念
                # 添加评价维度
                evaluation_words = ['评价', '对比', '留存', '满意度', '服务']
                for word in evaluation_words:
                    if word in top_words:
                        label += f'与{word}'
                        break
                # 确保标签长度合理
                if len(label) < 5:
                    label += '分析'
            else:
                # 如果没有匹配到核心概念，使用通用标签
                label = f'主题{topic_idx+1}分析'
            
            business_labels[topic_idx] = label
    else:
        #  fallback: 使用通用标签
        for i in range(n_topics):
            business_labels[i] = f'主题{i+1}分析'
        
    return business_labels

# 📋 5. 自动评估并选择最佳主题数
print("开始自动评估最佳主题数...")

def evaluate_topic_numbers(X_train, topic_numbers=[4, 5, 6], max_iter=200):
    """
    评估不同主题数的性能，选择最佳主题数
    
    Args:
        X_train: 训练集特征矩阵
        topic_numbers: 待评估的主题数列表
        max_iter: 迭代次数
        
    Returns:
        best_n_topics: 最佳主题数
        best_perplexity: 最低困惑度
        evaluation_results: 所有评估结果
    """
    evaluation_results = []
    
    for n_topics in topic_numbers:
        print(f"评估主题数: {n_topics}...")
        
        # 训练LDA模型
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=max_iter,
            learning_method='online',
            learning_offset=10.0,
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        lda.fit(X_train)
        
        # 计算困惑度
        perplexity = lda.perplexity(X_train)
        
        # 计算主题清晰度（主题词分布的熵）
        topic_entropy = []
        feature_names = vectorizer.get_feature_names_out()
        for topic in lda.components_:
            topic_dist = topic / topic.sum()
            entropy = -np.sum(topic_dist * np.log(topic_dist + 1e-10))
            topic_entropy.append(entropy)
        avg_topic_clarity = 1 - (np.mean(topic_entropy) / np.log(len(feature_names)))
        
        evaluation_results.append({
            'n_topics': n_topics,
            'perplexity': perplexity,
            'avg_topic_clarity': avg_topic_clarity
        })
        
        print(f"  困惑度: {perplexity:.2f}, 主题清晰度: {avg_topic_clarity:.4f}")
    
    # 选择最佳主题数（基于困惑度）
    best_result = min(evaluation_results, key=lambda x: x['perplexity'])
    best_n_topics = best_result['n_topics']
    best_perplexity = best_result['perplexity']
    
    print(f"\n最佳主题数: {best_n_topics}")
    print(f"最低困惑度: {best_perplexity:.2f}")
    
    return best_n_topics, best_perplexity, evaluation_results

# 评估最佳主题数
topic_numbers_to_evaluate = [4, 5, 6, 7]  # 扩展评估范围
n_topics, best_perplexity, evaluation_results = evaluate_topic_numbers(X_train_count, topic_numbers_to_evaluate)

# 获取业务标签（在LDA模型训练完成后调用）
# 注意：这里先设置一个默认值，后续会在LDA模型训练完成后更新
business_labels = get_business_labels(n_topics)
# 训练最终LDA模型
print(f"\n开始训练最终LDA模型（主题数={n_topics}）...")
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=200,
    learning_method='online',
    learning_offset=10.0,
    doc_topic_prior=0.1,
    topic_word_prior=0.01
)
lda.fit(X_train_count)
print("LDA模型训练完成")

# 基于训练好的LDA模型自动生成业务标签
print("基于关键词自动生成业务标签...")
business_labels = get_business_labels(n_topics, lda, vectorizer)
print("业务标签生成完成")
# 📋 6. 输出每个主题的关键词
def get_lda_keywords(model, vectorizer, business_labels, n_top_words=15):
    keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        topic_label = business_labels.get(topic_idx, f"主题 {topic_idx+1}")
        keywords[topic_label] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return keywords

def get_tfidf_keywords_by_topic(df, tfidf_vectorizer, business_labels, n_top_words=10):
    """
    基于TF-IDF为每个主题提取关键词
    
    Args:
        df: 包含文本和主题的数据框
        tfidf_vectorizer: TF-IDF向量化器
        business_labels: 业务标签字典
        n_top_words: 每个主题提取的关键词数量
        
    Returns:
        每个主题的TF-IDF关键词字典
    """
    topic_keywords = {}
    topics = df['dominant_topic'].unique()
    
    for topic in topics:
        # 获取该主题的所有文档
        topic_docs = df[df['dominant_topic'] == topic]['cleaned_text']
        
        if len(topic_docs) > 0:
            # 计算该主题所有文档的TF-IDF
            tfidf_matrix = tfidf_vectorizer.transform(topic_docs)
            # 计算每个词的平均TF-IDF值
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            # 获取特征名称
            feature_names = tfidf_vectorizer.get_feature_names_out()
            # 按TF-IDF值排序，取前n_top_words个
            top_indices = avg_tfidf.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_label = business_labels.get(topic, f"主题 {topic+1}")
            topic_keywords[topic_label] = top_words
    
    return topic_keywords

def get_combined_keywords(lda_keywords, tfidf_keywords, n_top_words=10):
    """
    结合LDA和TF-IDF关键词，确保两者都有贡献
    
    Args:
        lda_keywords: LDA主题关键词
        tfidf_keywords: TF-IDF主题关键词
        n_top_words: 每个主题保留的关键词数量
        
    Returns:
        每个主题的组合关键词字典
    """
    combined_keywords = {}
    
    for topic in lda_keywords:
        if topic in tfidf_keywords:
            # 合并两种关键词，去重，确保平衡融合
            combined = []
            seen = set()
            
            # 交替添加LDA和TF-IDF关键词，确保两者都有贡献
            lda_words = lda_keywords[topic]
            tfidf_words = tfidf_keywords[topic]
            
            max_len = max(len(lda_words), len(tfidf_words))
            
            for i in range(max_len):
                # 添加LDA关键词（如果有）
                if i < len(lda_words):
                    word = lda_words[i]
                    if word not in seen:
                        combined.append(word)
                        seen.add(word)
                        if len(combined) >= n_top_words:
                            break
                
                # 添加TF-IDF关键词（如果有）
                if i < len(tfidf_words):
                    word = tfidf_words[i]
                    if word not in seen:
                        combined.append(word)
                        seen.add(word)
                        if len(combined) >= n_top_words:
                            break
            
            # 如果还不够，添加剩余的LDA关键词
            if len(combined) < n_top_words:
                for word in lda_words:
                    if word not in seen:
                        combined.append(word)
                        seen.add(word)
                        if len(combined) >= n_top_words:
                            break
            
            # 如果还不够，添加剩余的TF-IDF关键词
            if len(combined) < n_top_words:
                for word in tfidf_words:
                    if word not in seen:
                        combined.append(word)
                        seen.add(word)
                        if len(combined) >= n_top_words:
                            break
            
            combined_keywords[topic] = combined
        else:
            combined_keywords[topic] = lda_keywords[topic][:n_top_words]
    
    return combined_keywords

# 提取关键词
keywords = get_lda_keywords(lda, vectorizer, business_labels)
print("\n自动生成的主题关键词：")
for theme, words in keywords.items():
    print(f"{theme}: {', '.join(words[:10])}")

# 📋 7. 为训练集分配主题
print("为训练集分配主题...")
doc_topic_dist_train = lda.transform(X_train_count)
df_train['dominant_topic'] = doc_topic_dist_train.argmax(axis=1)

# 📋 8. 准备TF-IDF特征（仅使用训练集）
print("创建TF-IDF向量化器（仅使用训练集）...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=8000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['cleaned_text'])
print(f"TF-IDF向量化完成，特征维度: {X_train_tfidf.shape[1]}")

# 📋 9. 提取每个主题的TF-IDF关键词
print("\n提取每个主题的TF-IDF关键词：")
tfidf_keywords = get_tfidf_keywords_by_topic(df_train, tfidf_vectorizer, business_labels)
for theme, words in tfidf_keywords.items():
    print(f"{theme} (TF-IDF): {', '.join(words[:10])}")

# 📋 10. 结合LDA和TF-IDF关键词
print("\n结合LDA和TF-IDF关键词：")
combined_keywords = get_combined_keywords(keywords, tfidf_keywords)
for theme, words in combined_keywords.items():
    print(f"{theme} (组合): {', '.join(words[:10])}")

# 📋 9. 为训练集合并特征
print("为训练集合并特征...")
X_combined_train = np.hstack([doc_topic_dist_train, X_train_tfidf.toarray()])
print(f"合并后特征维度: {X_combined_train.shape[1]}")

# 📋 10. 为验证集提取特征（使用训练集的模型）
print("为验证集提取特征...")
X_val_count = vectorizer.transform(df_val['cleaned_text'])
X_val_tfidf = tfidf_vectorizer.transform(df_val['cleaned_text'])
doc_topic_dist_val = lda.transform(X_val_count)
X_combined_val = np.hstack([doc_topic_dist_val, X_val_tfidf.toarray()])
df_val['dominant_topic'] = doc_topic_dist_val.argmax(axis=1)

# 📋 11. 为测试集提取特征（使用训练集的模型）
print("为测试集提取特征...")
X_test_count = vectorizer.transform(df_test['cleaned_text'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['cleaned_text'])
doc_topic_dist_test = lda.transform(X_test_count)
X_combined_test = np.hstack([doc_topic_dist_test, X_test_tfidf.toarray()])
df_test['dominant_topic'] = doc_topic_dist_test.argmax(axis=1)

# 📋 12. 训练主题分类器（改进参数）
print("开始训练主题分类器...")
# 使用改进的RandomForest参数
theme_classifier = RandomForestClassifier(
    n_estimators=100,        # 增加树的数量
    max_depth=50,            # 增加树的深度
    min_samples_split=5,     # 最小分裂样本数
    min_samples_leaf=2,      # 最小叶节点样本数
    random_state=42,
    n_jobs=-1                # 使用所有核心
)
theme_classifier.fit(X_combined_train, df_train['dominant_topic'])

# 📋 13. 评估模型
print("评估模型性能...")
# 在验证集上评估
y_val_pred = theme_classifier.predict(X_combined_val)
val_accuracy = accuracy_score(df_val['dominant_topic'], y_val_pred)
print(f"\n验证集准确率: {val_accuracy:.4f}")

# 在测试集上评估（最终评估）
y_test_pred = theme_classifier.predict(X_combined_test)
test_accuracy = accuracy_score(df_test['dominant_topic'], y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 打印详细评估报告（基于测试集）
print("\n分类报告（测试集）:")
target_names = [business_labels.get(i, f"主题 {i+1}") for i in range(n_topics)]
print(classification_report(df_test['dominant_topic'], y_test_pred, target_names=target_names))

# 计算困惑度（仅使用训练集）
perplexity = lda.perplexity(X_train_count)
print(f"\nLDA模型困惑度: {perplexity:.2f}")



# 📋 14. 保存模型和向量化器
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

# 📋 14. 生成训练报告
print("\n训练完成，生成训练报告...")
# 创建logs文件夹
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)

# 创建主题编号到业务标签的映射
topic_to_label = business_labels

# 创建业务标签到主题编号的反向映射
label_to_topic = {label: topic for topic, label in business_labels.items()}

with open(os.path.join(logs_dir, 'train_log_new.txt'), 'w', encoding='utf-8') as f:
    f.write("=== 模型训练报告 ===\n")
    f.write(f"训练时间: {pd.Timestamp.now()}\n")
    f.write(f"总数据量: {len(df)}\n")
    f.write(f"训练集: {len(df_train)} 条记录\n")
    f.write(f"验证集: {len(df_val)} 条记录\n")
    f.write(f"测试集: {len(df_test)} 条记录\n")
    f.write(f"特征维度: {X_train_count.shape[1]}\n")
    f.write(f"验证集准确率: {val_accuracy:.4f}\n")
    f.write(f"测试集准确率: {test_accuracy:.4f}\n")
    f.write(f"LDA困惑度: {perplexity:.2f}\n")
    f.write(f"最佳主题数: {n_topics}\n")
    f.write(f"主题数评估结果:\n")
    for result in evaluation_results:
        f.write(f"  主题数={result['n_topics']}: 困惑度={result['perplexity']:.2f}, 主题清晰度={result['avg_topic_clarity']:.4f}\n")
    f.write("\n")

    f.write("=== 主题业务标签 ===\n")
    for topic_id, label in business_labels.items():
        f.write(f"{label}\n")
    f.write("\n")

    f.write("=== LDA主题关键词 ===\n")
    for topic_idx, topic in enumerate(lda.components_):
        topic_label = business_labels.get(topic_idx, f"主题 {topic_idx+1}")
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        f.write(f"{topic_label}: {', '.join(top_words)}\n")
    
    f.write("\n=== TF-IDF主题关键词 ===\n")
    # 为每个主题生成TF-IDF关键词
    tfidf_keywords_by_label = {}
    for topic in df_train['dominant_topic'].unique():
        topic_docs = df_train[df_train['dominant_topic'] == topic]['cleaned_text']
        if len(topic_docs) > 0:
            tfidf_matrix = tfidf_vectorizer.transform(topic_docs)
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            feature_names = tfidf_vectorizer.get_feature_names_out()
            top_indices = avg_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_label = business_labels.get(topic, f"主题 {topic+1}")
            tfidf_keywords_by_label[topic_label] = top_words
    
    for label, words in tfidf_keywords_by_label.items():
        f.write(f"{label}: {', '.join(words)}\n")
    
    f.write("\n=== 组合主题关键词 ===\n")
    # 生成组合关键词
    combined_keywords_by_label = {}
    for topic_idx, topic in enumerate(lda.components_):
        topic_label = business_labels.get(topic_idx, f"主题 {topic_idx+1}")
        # 获取LDA关键词
        lda_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-16:-1]]
        # 获取TF-IDF关键词
        topic_docs = df_train[df_train['dominant_topic'] == topic_idx]['cleaned_text']
        tfidf_words = []
        if len(topic_docs) > 0:
            tfidf_matrix = tfidf_vectorizer.transform(topic_docs)
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            feature_names = tfidf_vectorizer.get_feature_names_out()
            top_indices = avg_tfidf.argsort()[-10:][::-1]
            tfidf_words = [feature_names[i] for i in top_indices]
        
        # 组合关键词
        combined = []
        seen = set()
        max_len = max(len(lda_words), len(tfidf_words))
        
        for i in range(max_len):
            if i < len(lda_words):
                word = lda_words[i]
                if word not in seen:
                    combined.append(word)
                    seen.add(word)
                    if len(combined) >= 10:
                        break
            if i < len(tfidf_words):
                word = tfidf_words[i]
                if word not in seen:
                    combined.append(word)
                    seen.add(word)
                    if len(combined) >= 10:
                        break
        
        if len(combined) < 10:
            for word in lda_words:
                if word not in seen:
                    combined.append(word)
                    seen.add(word)
                    if len(combined) >= 10:
                        break
        
        if len(combined) < 10:
            for word in tfidf_words:
                if word not in seen:
                    combined.append(word)
                    seen.add(word)
                    if len(combined) >= 10:
                        break
        
        combined_keywords_by_label[topic_label] = combined
    
    for label, words in combined_keywords_by_label.items():
        f.write(f"{label}: {', '.join(words)}\n")

print(f"训练报告已生成: {os.path.join(logs_dir, 'train_log_new.txt')}")

# 打印业务标签
print("\n主题业务标签：")
for theme, label in business_labels.items():
    print(f"{theme}: {label}")