#!/usr/bin/env python3
"""
主题质量评估与优化工具
计算主题间相似度，识别重叠主题，提供优化建议
"""

import os
import sys
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_preprocessor_enhanced import EnhancedTextPreprocessor
from data_preprocessor import DataPreprocessor
from topic_modeler import TopicModeler
from label_generator import LabelGenerator
from keyword_extractor import KeywordExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/topic_quality_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TopicQualityAnalyzer:
    """主题质量分析器"""
    
    def __init__(self, data_file):
        """
        初始化主题质量分析器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.topic_modeler = None
        self.keyword_extractor = None
        self.topics = []
        self.topic_keywords = {}
    
    def load_model(self):
        """加载模型和数据"""
        try:
            logger.info("加载数据和模型...")
            
            # 初始化预处理器
            preprocessor = EnhancedTextPreprocessor()
            
            # 初始化数据预处理器
            data_preprocessor = DataPreprocessor(preprocessor)
            
            # 加载和预处理数据
            df = data_preprocessor.load_data()
            df = data_preprocessor.preprocess_data(df)
            
            # 划分数据
            df_train, df_val, df_test = data_preprocessor.split_data(df)
            
            # 提取特征和标签
            X_train = df_train['cleaned_text'].tolist()
            X_val = df_val['cleaned_text'].tolist()
            X_test = df_test['cleaned_text'].tolist()
            y_train = [0] * len(X_train)  # 临时标签，实际使用时会被主题分配替换
            y_val = [0] * len(X_val)
            y_test = [0] * len(X_test)
            
            # 初始化主题模型
            self.topic_modeler = TopicModeler()
            
            # 训练LDA模型
            best_num_topics = self.topic_modeler.evaluate_topic_numbers(X_train)
            self.topic_modeler.train_final_model(X_train, best_num_topics)
            
            # 生成业务标签
            label_generator = LabelGenerator()
            business_labels = label_generator.get_business_labels(
                best_num_topics, 
                self.topic_modeler.lda, 
                self.topic_modeler.count_vectorizer
            )
            
            # 初始化关键词提取器
            self.keyword_extractor = KeywordExtractor()
            
            # 提取主题关键词
            self.topic_names = list(business_labels.values())
            self.topic_keywords = self.keyword_extractor.get_lda_keywords(
                self.topic_modeler.lda,
                self.topic_modeler.count_vectorizer,
                business_labels
            )
            
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise
    
    def calculate_topic_similarity(self):
        """
        计算主题间的相似度矩阵
        
        Returns:
            np.ndarray: 主题相似度矩阵
        """
        try:
            logger.info("计算主题间相似度...")
            
            # 提取主题-词分布
            topic_word_dist = self.topic_modeler.lda.components_
            
            # 归一化主题分布
            topic_word_dist_normalized = topic_word_dist / topic_word_dist.sum(axis=1, keepdims=True)
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(topic_word_dist_normalized)
            
            logger.info("主题相似度计算完成")
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"计算主题相似度时出错: {e}")
            raise
    
    def identify_overlapping_topics(self, similarity_matrix, threshold=0.7):
        """
        识别高度重叠的主题对
        
        Args:
            similarity_matrix: 主题相似度矩阵
            threshold: 相似度阈值
            
        Returns:
            list: 高度重叠的主题对
        """
        try:
            logger.info(f"识别相似度高于 {threshold} 的主题对...")
            
            overlapping_pairs = []
            
            # 遍历相似度矩阵
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > threshold:
                        overlapping_pairs.append({
                            'topic1': i,
                            'topic1_name': self.topic_names[i],
                            'topic2': j,
                            'topic2_name': self.topic_names[j],
                            'similarity': similarity_matrix[i, j]
                        })
            
            # 按相似度排序
            overlapping_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"识别出 {len(overlapping_pairs)} 对高度重叠的主题")
            
            return overlapping_pairs
            
        except Exception as e:
            logger.error(f"识别重叠主题时出错: {e}")
            raise
    
    def analyze_topic_quality(self):
        """
        分析主题质量
        
        Returns:
            dict: 主题质量分析结果
        """
        try:
            logger.info("开始主题质量分析...")
            
            # 加载模型
            self.load_model()
            
            # 计算主题相似度
            similarity_matrix = self.calculate_topic_similarity()
            
            # 识别重叠主题
            overlapping_pairs = self.identify_overlapping_topics(similarity_matrix)
            
            # 分析主题关键词质量
            keyword_quality = self.analyze_keyword_quality()
            
            # 生成分析报告
            analysis_result = {
                'topic_count': len(self.topic_keywords),
                'topic_names': list(self.topic_keywords.keys()),
                'similarity_matrix': similarity_matrix.tolist(),
                'overlapping_pairs': overlapping_pairs,
                'keyword_quality': keyword_quality,
                'recommendations': self.generate_recommendations(overlapping_pairs, keyword_quality)
            }
            
            logger.info("主题质量分析完成")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"分析主题质量时出错: {e}")
            raise
    
    def analyze_keyword_quality(self):
        """
        分析关键词质量

        Returns:
            dict: 关键词质量分析结果
        """
        keyword_quality = []
        
        for i, (topic_name, keywords) in enumerate(self.topic_keywords.items()):
            # 分析关键词质量
            duplicate_phrases = [kw for kw in keywords if ' ' in kw]
            vague_keywords = [kw for kw in keywords if kw in ['这种', '然后', '这样', '那样']]
            meaningful_keywords = [kw for kw in keywords if kw not in ['这种', '然后', '这样', '那样'] and ' ' not in kw]
            
            keyword_quality.append({
                'topic_id': i,
                'topic_name': topic_name,
                'total_keywords': len(keywords),
                'meaningful_keywords': len(meaningful_keywords),
                'duplicate_phrases': len(duplicate_phrases),
                'vague_keywords': len(vague_keywords),
                'keywords': keywords
            })
        
        return keyword_quality
    
    def generate_recommendations(self, overlapping_pairs, keyword_quality):
        """
        生成优化建议
        
        Args:
            overlapping_pairs: 重叠主题对
            keyword_quality: 关键词质量分析
            
        Returns:
            list: 优化建议
        """
        recommendations = []
        
        # 主题合并建议
        if overlapping_pairs:
            recommendations.append({
                'type': 'merge',
                'description': '合并高度重叠的主题',
                'details': overlapping_pairs
            })
        
        # 关键词优化建议
        vague_topics = [item for item in keyword_quality if item['vague_keywords'] > 0]
        if vague_topics:
            recommendations.append({
                'type': 'keyword_optimization',
                'description': '优化关键词质量，移除模糊词汇',
                'details': vague_topics
            })
        
        # 主题命名优化建议
        unclear_names = [name for name in self.topic_keywords.keys() if '相关分析' in name or '好像' in name]
        if unclear_names:
            recommendations.append({
                'type': 'naming_optimization',
                'description': '优化主题命名，使用更具描述性的名称',
                'details': unclear_names
            })
        
        # 主题数量调整建议
        if len(self.topic_keywords) > 5:
            recommendations.append({
                'type': 'topic_count',
                'description': '考虑减少主题数量，避免主题碎片化',
                'details': f'当前主题数: {len(self.topic_keywords)}'
            })
        
        return recommendations
    
    def save_analysis(self, analysis_result):
        """
        保存分析结果
        
        Args:
            analysis_result: 分析结果
        """
        import json
        from datetime import datetime
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/topic_quality_analysis_{timestamp}.json'
        
        # 确保logs目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 保存结果
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {filename}")

if __name__ == "__main__":
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='主题质量分析')
    parser.add_argument('--data', type=str, default='../data/processed_data.csv',
                        help='数据文件路径')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = TopicQualityAnalyzer(args.data)
    
    # 执行分析
    analysis_result = analyzer.analyze_topic_quality()
    
    # 保存结果
    analyzer.save_analysis(analysis_result)
    
    # 打印结果
    print("主题质量分析结果:")
    print(f"主题数量: {analysis_result['topic_count']}")
    print(f"主题名称: {analysis_result['topic_names']}")
    print(f"重叠主题对数量: {len(analysis_result['overlapping_pairs'])}")
    print(f"优化建议数量: {len(analysis_result['recommendations'])}")
    
    print("\n优化建议:")
    for i, recommendation in enumerate(analysis_result['recommendations'], 1):
        print(f"{i}. {recommendation['description']}")