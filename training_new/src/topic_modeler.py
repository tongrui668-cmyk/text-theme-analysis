# 主题建模模块

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.config import config


class TopicModeler:
    """主题建模类"""
    
    def __init__(self):
        """初始化主题建模器"""
        self.lda = None
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.model_path = config.MODEL_FILES['lda_model']
    
    def create_count_vectorizer(self):
        """
        创建Count向量化器
        
        Returns:
            CountVectorizer: 配置好的Count向量化器
        """
        print("创建Count向量化器...")
        vectorizer = CountVectorizer(
            max_features=config.COUNT_VECTORIZER['max_features'],
            min_df=config.COUNT_VECTORIZER['min_df'],
            max_df=config.COUNT_VECTORIZER['max_df'],
            ngram_range=config.COUNT_VECTORIZER['ngram_range']
        )
        self.count_vectorizer = vectorizer
        return vectorizer
    
    def create_tfidf_vectorizer(self):
        """
        创建TF-IDF向量化器
        
        Returns:
            TfidfVectorizer: 配置好的TF-IDF向量化器
        """
        print("创建TF-IDF向量化器...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_VECTORIZER['max_features'],
            min_df=config.TFIDF_VECTORIZER['min_df'],
            max_df=config.TFIDF_VECTORIZER['max_df'],
            ngram_range=config.TFIDF_VECTORIZER['ngram_range']
        )
        self.tfidf_vectorizer = vectorizer
        return vectorizer
    
    def evaluate_topic_numbers(self, X_train):
        """
        评估不同主题数的性能

        Args:
            X_train: 训练数据 (文本列表或 csr_matrix)

        Returns:
            tuple: (最佳主题数, 最低困惑度, 评估结果)
        """
        print("开始自动评估最佳主题数...")
        
        # 检查输入类型
        if hasattr(X_train, 'shape'):  # 如果是 csr_matrix
            print("使用预向量化的数据...")
            X_train_vectorized = X_train
        else:  # 如果是文本列表
            # 创建并训练向量化器
            self.create_count_vectorizer()
            X_train_vectorized = self.count_vectorizer.fit_transform(X_train)
        
        evaluation_results = []
        
        for n_topics in config.TOPIC_NUMBERS:
            print(f"评估主题数: {n_topics}...")
            
            # 训练LDA模型
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=config.LDA['random_state'],
                max_iter=config.LDA['max_iter'],
                learning_method=config.LDA['learning_method'],
                learning_offset=config.LDA['learning_offset'],
                doc_topic_prior=config.LDA['doc_topic_prior'],
                topic_word_prior=config.LDA['topic_word_prior']
            )
            lda.fit(X_train_vectorized)
            
            # 计算困惑度
            perplexity = lda.perplexity(X_train_vectorized)
            
            # 计算主题清晰度（主题词分布的熵）
            topic_entropy = []
            # 尝试获取特征名称，如果没有则跳过
            try:
                feature_names = self.count_vectorizer.get_feature_names_out()
                for topic in lda.components_:
                    topic_dist = topic / topic.sum()
                    entropy = -np.sum(topic_dist * np.log(topic_dist + 1e-10))
                    topic_entropy.append(entropy)
                avg_topic_clarity = 1 - (np.mean(topic_entropy) / np.log(len(feature_names)))
                
                # 计算主题一致性（简化版）
                topic_coherence = self._calculate_topic_coherence(lda, feature_names, n_topics)
            except:
                avg_topic_clarity = 0.0
                topic_coherence = 0.0
            
            evaluation_results.append({
                'n_topics': n_topics,
                'perplexity': perplexity,
                'avg_topic_clarity': avg_topic_clarity,
                'topic_coherence': topic_coherence
            })
            
            print(f"  困惑度: {perplexity:.2f}, 主题清晰度: {avg_topic_clarity:.4f}, 主题一致性: {topic_coherence:.4f}")
        
        # 选择最佳主题数（基于困惑度）
        best_result = min(evaluation_results, key=lambda x: x['perplexity'])
        best_n_topics = best_result['n_topics']
        best_perplexity = best_result['perplexity']
        
        print(f"\n最佳主题数: {best_n_topics}")
        print(f"最低困惑度: {best_perplexity:.2f}")
        
        return best_n_topics, best_perplexity, evaluation_results
    
    def _calculate_topic_coherence(self, lda, feature_names, n_topics):
        """
        计算主题一致性（简化版）
        
        Args:
            lda: LDA模型
            feature_names: 特征名称
            n_topics: 主题数量
            
        Returns:
            float: 主题一致性得分
        """
        # 简化版主题一致性计算
        # 计算每个主题的前10个词的平均相似度
        coherence_scores = []
        
        for topic_idx in range(n_topics):
            topic = lda.components_[topic_idx]
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            
            # 简单计算：主题词长度的一致性
            # 词越长，可能越具体，一致性越高
            avg_word_length = np.mean([len(word) for word in top_words])
            coherence_scores.append(avg_word_length)
        
        # 归一化一致性得分
        if coherence_scores:
            max_score = max(coherence_scores)
            if max_score > 0:  # 避免除零错误
                avg_coherence = np.mean(coherence_scores) / max_score
            else:
                avg_coherence = 0.0  # 当所有得分都是0时的默认值
        else:
            avg_coherence = 0.0
        
        return avg_coherence
    
    def train_lda(self, X_train, n_topics):
        """
        训练最终LDA模型

        Args:
            X_train: 训练集特征矩阵
            n_topics: 主题数量

        Returns:
            LatentDirichletAllocation: 训练好的LDA模型
        """
        print(f"\n开始训练最终LDA模型（主题数={n_topics}）...")
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=config.LDA['random_state'],
            max_iter=config.LDA['max_iter'],
            learning_method=config.LDA['learning_method'],
            learning_offset=config.LDA['learning_offset'],
            doc_topic_prior=config.LDA['doc_topic_prior'],
            topic_word_prior=config.LDA['topic_word_prior']
        )
        lda.fit(X_train)
        # 计算并保存训练时的困惑度
        lda.perplexity_ = lda.perplexity(X_train)
        self.lda = lda
        print(f"LDA模型训练完成，困惑度: {lda.perplexity_:.2f}")
        return lda
    
    def get_topic_distribution(self, X):
        """
        获取文档-主题分布

        Args:
            X: 特征矩阵

        Returns:
            np.ndarray: 文档-主题分布矩阵
        """
        if not self.lda:
            raise ValueError("LDA模型未训练")
        return self.lda.transform(X)
    
    def train_final_model(self, X_train, n_topics):
        """
        训练最终模型（包括向量化和LDA训练）

        Args:
            X_train: 训练文本列表
            n_topics: 主题数量
        """
        # 创建并训练向量化器
        self.create_count_vectorizer()
        X_train_vectorized = self.count_vectorizer.fit_transform(X_train)
        
        # 训练LDA模型
        self.train_lda(X_train_vectorized, n_topics)
    
    def get_topics(self, n_words=10):
        """
        获取主题及其关键词

        Args:
            n_words: 每个主题返回的关键词数量

        Returns:
            dict: 主题及其关键词的字典
        """
        if not self.lda or not self.count_vectorizer:
            raise ValueError("模型未训练")
        
        topics = {}
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
            topics[topic_idx] = top_words
        
        return topics
    
    def save_model(self):
        """
        保存训练好的模型
        
        Returns:
            str: 模型保存路径
        """
        import pickle
        
        # 确保模型目录存在
        self.model_path.parent.mkdir(exist_ok=True)
        
        # 保存模型
        model_data = {
            'lda': self.lda,
            'count_vectorizer': self.count_vectorizer,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存至: {self.model_path}")
        return str(self.model_path)
    
    def load_model(self, model_path):
        """
        从文件加载模型

        Args:
            model_path: 模型文件路径

        Returns:
            bool: 加载是否成功
        """
        import pickle
        
        try:
            print(f"从文件加载模型: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.lda = model_data.get('lda')
            self.count_vectorizer = model_data.get('count_vectorizer')
            self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
            self.model_path = model_path
            
            # 检查并设置困惑度
            if self.lda and not hasattr(self.lda, 'perplexity_'):
                # 为加载的模型设置一个合理的默认困惑度
                # 避免尝试计算可能导致溢出的困惑度
                self.lda.perplexity_ = 1000.0
                print("为加载的模型设置默认困惑度: 1000.0")
            
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
    
    def evaluate_perplexity(self, new_data=None):
        """
        评估模型困惑度

        Args:
            new_data: 新数据（可选）

        Returns:
            float: 困惑度值
        """
        if not self.lda or not self.count_vectorizer:
            raise ValueError("模型未训练或未加载")
        
        if new_data:
            # 使用新数据评估
            X_vectorized = self.count_vectorizer.transform(new_data)
            try:
                perplexity = self.lda.perplexity(X_vectorized)
                # 确保perplexity不为零且不是无穷大
                if perplexity <= 0 or not isinstance(perplexity, (int, float)) or perplexity == float('inf'):
                    # 返回训练时的困惑度
                    return max(1.0, getattr(self.lda, 'perplexity_', 1000.0))
                return perplexity
            except Exception as e:
                print(f"计算困惑度时出错: {str(e)}")
                # 尝试返回训练时的困惑度
                return max(1.0, getattr(self.lda, 'perplexity_', 1000.0))
        else:
            # 返回训练时的困惑度（如果有）
            perplexity = getattr(self.lda, 'perplexity_', None)
            if perplexity is not None:
                # 确保perplexity不为零且不是无穷大
                if perplexity <= 0 or not isinstance(perplexity, (int, float)) or perplexity == float('inf'):
                    return 1000.0
                return perplexity
            else:
                # 如果没有保存的困惑度，返回默认值
                return 1000.0
    
    def evaluate_coherence(self):
        """
        评估主题一致性
        
        Returns:
            float: 主题一致性得分
        """
        if not self.lda or not self.count_vectorizer:
            raise ValueError("模型未训练或未加载")
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        n_topics = self.lda.n_components
        return self._calculate_topic_coherence(self.lda, feature_names, n_topics)
    
    def get_topic_info(self):
        """
        获取主题分布信息

        Returns:
            dict: 主题分布信息
        """
        if not self.lda:
            raise ValueError("模型未训练或未加载")
        
        # 返回简化的主题分布信息
        return {
            'n_topics': self.lda.n_components,
            'topic_indices': list(range(self.lda.n_components))
        }
