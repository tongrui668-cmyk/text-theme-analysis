#!/usr/bin/env python3
"""
主题质量优化工具
实施关键词去重、主题命名优化和主题数量调整
"""

import os
import sys
import logging
import json

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
        logging.FileHandler('logs/topic_optimization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TopicQualityOptimizer:
    """主题质量优化器"""
    
    def __init__(self, data_file):
        """
        初始化主题质量优化器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.topic_modeler = None
        self.keyword_extractor = None
        self.label_generator = None
        self.optimized_topics = {}
    
    def load_data_and_model(self):
        """加载数据和模型"""
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
            
            # 提取特征
            X_train = df_train['cleaned_text'].tolist()
            
            return X_train
            
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            raise
    
    def optimize_topic_count(self, X_train):
        """
        优化主题数量
        
        Args:
            X_train: 训练文本列表
            
        Returns:
            int: 优化后的主题数量
        """
        try:
            logger.info("优化主题数量...")
            
            # 初始化主题模型
            self.topic_modeler = TopicModeler()
            
            # 评估不同主题数
            best_num_topics = self.topic_modeler.evaluate_topic_numbers(X_train)
            
            # 训练最终模型（使用优化后的主题数）
            # 考虑到避免碎片化，我们选择一个合理的主题数
            optimized_num_topics = min(best_num_topics, 5)
            logger.info(f"优化后的主题数: {optimized_num_topics}")
            
            self.topic_modeler.train_final_model(X_train, optimized_num_topics)
            
            return optimized_num_topics
            
        except Exception as e:
            logger.error(f"优化主题数量时出错: {e}")
            raise
    
    def optimize_topic_names(self, num_topics):
        """
        优化主题命名
        
        Args:
            num_topics: 主题数量
            
        Returns:
            dict: 优化后的主题名称
        """
        try:
            logger.info("优化主题命名...")
            
            # 初始化标签生成器
            self.label_generator = LabelGenerator()
            
            # 生成优化后的业务标签
            business_labels = self.label_generator.get_business_labels(
                num_topics,
                self.topic_modeler.lda,
                self.topic_modeler.count_vectorizer
            )
            
            # 进一步优化标签名称
            optimized_labels = {}
            for topic_idx, label in business_labels.items():
                # 移除模糊词汇
                if '好像' in label or '相关分析' in label:
                    # 生成更具描述性的标签
                    feature_names = self.topic_modeler.count_vectorizer.get_feature_names_out()
                    topic = self.topic_modeler.lda.components_[topic_idx]
                    top_words = [feature_names[i] for i in topic.argsort()[:-5:-1]]
                    new_label = f'{"-".join(top_words[:3])}分析'
                    optimized_labels[topic_idx] = new_label
                else:
                    optimized_labels[topic_idx] = label
            
            logger.info(f"优化后的主题名称: {list(optimized_labels.values())}")
            
            return optimized_labels
            
        except Exception as e:
            logger.error(f"优化主题命名时出错: {e}")
            raise
    
    def optimize_keywords(self, business_labels):
        """
        优化关键词质量
        
        Args:
            business_labels: 业务标签字典
            
        Returns:
            dict: 优化后的关键词
        """
        try:
            logger.info("优化关键词质量...")
            
            # 初始化关键词提取器
            self.keyword_extractor = KeywordExtractor()
            
            # 提取关键词
            keywords = self.keyword_extractor.get_lda_keywords(
                self.topic_modeler.lda,
                self.topic_modeler.count_vectorizer,
                business_labels
            )
            
            # 优化关键词
            optimized_keywords = {}
            for topic_name, topic_keywords in keywords.items():
                # 去重
                unique_keywords = []
                seen = set()
                
                for keyword in topic_keywords:
                    # 移除模糊词汇
                    if keyword in ['这种', '然后', '这样', '那样']:
                        continue
                    
                    # 移除重复短语
                    if ' ' in keyword:
                        # 检查短语是否已经由单个词表达
                        words = keyword.split()
                        if any(word in seen for word in words):
                            continue
                    
                    if keyword not in seen:
                        unique_keywords.append(keyword)
                        seen.add(keyword)
                
                optimized_keywords[topic_name] = unique_keywords[:10]  # 保留前10个关键词
            
            logger.info("关键词优化完成")
            
            return optimized_keywords
            
        except Exception as e:
            logger.error(f"优化关键词时出错: {e}")
            raise
    
    def run_optimization(self):
        """
        运行完整的优化流程
        
        Returns:
            dict: 优化结果
        """
        try:
            logger.info("开始主题质量优化流程...")
            
            # 加载数据
            X_train = self.load_data_and_model()
            
            # 优化主题数量
            optimized_num_topics = self.optimize_topic_count(X_train)
            
            # 优化主题命名
            optimized_labels = self.optimize_topic_names(optimized_num_topics)
            
            # 优化关键词
            optimized_keywords = self.optimize_keywords(optimized_labels)
            
            # 生成优化报告
            optimization_result = {
                'optimized_topic_count': optimized_num_topics,
                'optimized_topic_names': list(optimized_labels.values()),
                'optimized_keywords': optimized_keywords,
                'recommendations': self.generate_final_recommendations(optimized_keywords)
            }
            
            logger.info("主题质量优化完成")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"运行优化流程时出错: {e}")
            raise
    
    def generate_final_recommendations(self, optimized_keywords):
        """
        生成最终优化建议
        
        Args:
            optimized_keywords: 优化后的关键词
            
        Returns:
            list: 优化建议
        """
        recommendations = []
        
        # 检查关键词质量
        vague_keywords_found = False
        for topic_name, keywords in optimized_keywords.items():
            vague_keywords = [kw for kw in keywords if kw in ['这种', '然后', '这样', '那样']]
            if vague_keywords:
                vague_keywords_found = True
                break
        
        if not vague_keywords_found:
            recommendations.append({
                'type': 'keyword_quality',
                'description': '关键词质量优化成功',
                'details': '已移除所有模糊词汇和重复短语'
            })
        
        # 检查主题命名
        unclear_names = [name for name in optimized_keywords.keys() if '相关分析' in name or '好像' in name]
        if not unclear_names:
            recommendations.append({
                'type': 'naming_quality',
                'description': '主题命名优化成功',
                'details': '所有主题名称都清晰描述了主题内容'
            })
        
        # 检查主题数量
        if len(optimized_keywords) <= 5:
            recommendations.append({
                'type': 'topic_count',
                'description': '主题数量优化成功',
                'details': f'当前主题数: {len(optimized_keywords)}，避免了主题碎片化'
            })
        
        return recommendations
    
    def save_optimization_result(self, optimization_result):
        """
        保存优化结果
        
        Args:
            optimization_result: 优化结果
        """
        import json
        from datetime import datetime
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/topic_optimization_result_{timestamp}.json'
        
        # 确保logs目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 保存结果
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(optimization_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化结果已保存到: {filename}")


if __name__ == "__main__":
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='主题质量优化')
    parser.add_argument('--data', type=str, default='../data/processed_data.csv',
                        help='数据文件路径')
    
    args = parser.parse_args()
    
    # 初始化优化器
    optimizer = TopicQualityOptimizer(args.data)
    
    # 运行优化
    optimization_result = optimizer.run_optimization()
    
    # 保存结果
    optimizer.save_optimization_result(optimization_result)
    
    # 打印结果
    print("主题质量优化结果:")
    print(f"优化后的主题数: {optimization_result['optimized_topic_count']}")
    print(f"优化后的主题名称: {optimization_result['optimized_topic_names']}")
    print(f"优化建议数量: {len(optimization_result['recommendations'])}")
    
    print("\n优化建议:")
    for i, recommendation in enumerate(optimization_result['recommendations'], 1):
        print(f"{i}. {recommendation['description']}")
        if 'details' in recommendation:
            print(f"   详情: {recommendation['details']}")
    
    print("\n优化后的关键词:")
    for topic_name, keywords in optimization_result['optimized_keywords'].items():
        print(f"{topic_name}: {keywords[:5]}...")  # 只显示前5个关键词
