# 关键词提取模块

from src.config import config


class KeywordExtractor:
    """关键词提取类"""
    
    def __init__(self):
        """初始化关键词提取器"""
        pass
    
    def get_lda_keywords(self, model, vectorizer, business_labels, n_top_words=None):
        """
        从LDA模型提取关键词
        
        Args:
            model: LDA模型
            vectorizer: 向量化器
            business_labels: 业务标签字典
            n_top_words: 提取的关键词数量
            
        Returns:
            dict: 每个主题的关键词
        """
        if n_top_words is None:
            n_top_words = config.KEYWORDS['lda_top_words']
        
        keywords = {}
        feature_names = vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(model.components_):
            topic_label = business_labels.get(topic_idx, f"主题 {topic_idx+1}")
            keywords[topic_label] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        
        return keywords
    
    def get_tfidf_keywords_by_topic(self, df, tfidf_vectorizer, business_labels, n_top_words=None):
        """
        基于TF-IDF为每个主题提取关键词
        
        Args:
            df: 包含文本和主题的数据框
            tfidf_vectorizer: TF-IDF向量化器
            business_labels: 业务标签字典
            n_top_words: 提取的关键词数量
            
        Returns:
            dict: 每个主题的TF-IDF关键词
        """
        if n_top_words is None:
            n_top_words = config.KEYWORDS['tfidf_top_words']
        
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
    
    def get_combined_keywords(self, lda_keywords, tfidf_keywords, n_top_words=None):
        """
        结合LDA和TF-IDF关键词
        
        Args:
            lda_keywords: LDA主题关键词
            tfidf_keywords: TF-IDF主题关键词
            n_top_words: 每个主题保留的关键词数量
            
        Returns:
            dict: 每个主题的组合关键词
        """
        if n_top_words is None:
            n_top_words = config.KEYWORDS['combined_top_words']
        
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
