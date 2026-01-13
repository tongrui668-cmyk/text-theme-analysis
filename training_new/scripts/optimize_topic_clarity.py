#!/usr/bin/env python3
# 主题清晰度优化脚本

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_preprocessor import DataPreprocessor
from src.topic_modeler import TopicModeler
from src.text_preprocessor_enhanced import EnhancedTextPreprocessor


class TopicClarityOptimizer:
    """主题清晰度优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.text_preprocessor = EnhancedTextPreprocessor()
        self.data_preprocessor = DataPreprocessor(self.text_preprocessor)
        self.topic_modeler = TopicModeler()
    
    def load_data(self):
        """加载和预处理数据"""
        print("加载和预处理数据...")
        df = self.data_preprocessor.load_data()
        df = self.data_preprocessor.preprocess_data(df)
        df_train, _, _ = self.data_preprocessor.split_data(df)
        return df_train
    
    def optimize_topic_clarity(self):
        """优化主题清晰度"""
        print("开始优化主题清晰度...")
        
        # 加载数据
        df_train = self.load_data()
        
        # 准备训练数据
        vectorizer = self.topic_modeler.create_count_vectorizer()
        X_train_count = vectorizer.fit_transform(df_train['cleaned_text'])
        print(f"向量化完成，特征维度: {X_train_count.shape[1]}")
        
        # 定义参数网格
        param_grid = {
            'topic_word_prior': [0.0001, 0.001, 0.01, 0.1],
            'doc_topic_prior': [0.1, 0.3, 0.5],
            'max_iter': [500, 800, 1000],
            'n_topics': [5, 6, 7, 8]
        }
        
        best_clarity = 0.0
        best_params = {}
        
        # 遍历参数组合
        total_combinations = (
            len(param_grid['topic_word_prior']) *
            len(param_grid['doc_topic_prior']) *
            len(param_grid['max_iter']) *
            len(param_grid['n_topics'])
        )
        print(f"总参数组合数: {total_combinations}")
        
        current = 0
        for topic_word_prior in param_grid['topic_word_prior']:
            for doc_topic_prior in param_grid['doc_topic_prior']:
                for max_iter in param_grid['max_iter']:
                    for n_topics in param_grid['n_topics']:
                        current += 1
                        print(f"\n测试参数组合 {current}/{total_combinations}:")
                        print(f"topic_word_prior={topic_word_prior}, doc_topic_prior={doc_topic_prior}, max_iter={max_iter}, n_topics={n_topics}")
                        
                        # 创建并训练LDA模型
                        from sklearn.decomposition import LatentDirichletAllocation
                        from src.config import config
                        
                        lda = LatentDirichletAllocation(
                            n_components=n_topics,
                            random_state=config.LDA['random_state'],
                            max_iter=max_iter,
                            learning_method=config.LDA['learning_method'],
                            learning_offset=config.LDA['learning_offset'],
                            doc_topic_prior=doc_topic_prior,
                            topic_word_prior=topic_word_prior
                        )
                        
                        try:
                            lda.fit(X_train_count)
                            
                            # 计算主题清晰度
                            topic_entropy = []
                            feature_names = vectorizer.get_feature_names_out()
                            for topic in lda.components_:
                                topic_dist = topic / topic.sum()
                                entropy = -np.sum(topic_dist * np.log(topic_dist + 1e-10))
                                topic_entropy.append(entropy)
                            avg_topic_clarity = 1 - (np.mean(topic_entropy) / np.log(len(feature_names)))
                            
                            # 计算困惑度
                            perplexity = lda.perplexity(X_train_count)
                            
                            print(f"  主题清晰度: {avg_topic_clarity:.4f}, 困惑度: {perplexity:.2f}")
                            
                            # 更新最佳参数
                            if avg_topic_clarity > best_clarity:
                                best_clarity = avg_topic_clarity
                                best_params = {
                                    'topic_word_prior': topic_word_prior,
                                    'doc_topic_prior': doc_topic_prior,
                                    'max_iter': max_iter,
                                    'n_topics': n_topics,
                                    'clarity': avg_topic_clarity,
                                    'perplexity': perplexity
                                }
                                print(f"  🔍 发现更佳参数组合!")
                                
                        except Exception as e:
                            print(f"  ❌ 训练失败: {str(e)}")
                            continue
        
        # 输出最佳参数
        print("\n" + "="*60)
        print("最佳参数组合:")
        print("="*60)
        for key, value in best_params.items():
            print(f"{key}: {value}")
        print("="*60)
        
        return best_params


if __name__ == "__main__":
    optimizer = TopicClarityOptimizer()
    best_params = optimizer.optimize_topic_clarity()
    print("\n优化完成!")