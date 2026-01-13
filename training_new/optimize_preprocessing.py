#!/usr/bin/env python3
"""
预处理参数调优模块
基于验证集性能动态调整预处理参数
"""

import os
import sys
import logging
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_preprocessor_enhanced import EnhancedTextPreprocessor
from data_preprocessor import DataPreprocessor
from topic_modeler import TopicModeler
from label_generator import LabelGenerator
from keyword_extractor import KeywordExtractor
from classifier import Classifier
from model_saver import ModelSaver
from report_generator import ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimize_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PreprocessingOptimizer:
    """预处理参数调优器"""
    
    def __init__(self, data_file):
        """
        初始化调优器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.best_params = None
        self.best_metrics = {
            'accuracy': 0,
            'f1_score': 0,
            'perplexity': float('inf'),
            'topic_clarity': 0
        }
        self.param_combinations = self._generate_param_combinations()
    
    def _generate_param_combinations(self):
        """
        生成参数组合
        
        Returns:
            list: 参数组合列表
        """
        # 生成不同的预处理参数组合
        param_combinations = []
        
        # 测试不同的语义增强策略
        semantic_strategies = [
            'full',  # 完整语义增强
            'moderate',  # 适度语义增强
            'minimal'  # 最小语义增强
        ]
        
        # 测试不同的停用词策略
        stopword_strategies = [
            'full',  # 完整停用词
            'moderate',  # 适度停用词
            'minimal'  # 最小停用词
        ]
        
        # 测试不同的噪声过滤策略
        noise_filter_strategies = [
            'full',  # 完整噪声过滤
            'moderate',  # 适度噪声过滤
            'minimal'  # 最小噪声过滤
        ]
        
        # 生成参数组合
        for semantic_strategy in semantic_strategies:
            for stopword_strategy in stopword_strategies:
                for noise_filter_strategy in noise_filter_strategies:
                    param_combinations.append({
                        'semantic_strategy': semantic_strategy,
                        'stopword_strategy': stopword_strategy,
                        'noise_filter_strategy': noise_filter_strategy
                    })
        
        return param_combinations
    
    def _get_preprocessor_with_params(self, params):
        """
        根据参数获取预处理器
        
        Args:
            params: 预处理参数
            
        Returns:
            EnhancedTextPreprocessor: 配置好的预处理器
        """
        # 创建预处理器
        preprocessor = EnhancedTextPreprocessor()
        
        # 根据参数调整预处理器
        # 这里可以根据需要修改预处理器的行为
        # 例如：调整停用词列表、语义增强策略等
        
        return preprocessor
    
    def evaluate_params(self, params):
        """
        评估特定参数组合的性能
        
        Args:
            params: 预处理参数
            
        Returns:
            dict: 性能指标
        """
        try:
            logger.info(f"评估参数组合: {params}")
            
            # 获取配置好的预处理器
            preprocessor = self._get_preprocessor_with_params(params)
            
            # 初始化数据预处理器
            data_preprocessor = DataPreprocessor(preprocessor)
            
            # 加载和预处理数据
            data_preprocessor.load_data(self.data_file)
            data_preprocessor.preprocess_data()
            
            # 划分数据
            X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.split_data()
            
            # 初始化主题模型
            topic_modeler = TopicModeler()
            
            # 评估主题数
            best_num_topics = topic_modeler.evaluate_topic_numbers(X_train)
            
            # 训练最终模型
            topic_modeler.train_final_model(X_train, best_num_topics)
            
            # 生成业务标签
            label_generator = LabelGenerator(topic_modeler)
            business_labels = label_generator.generate_business_labels()
            
            # 分配主题
            train_topics = topic_modeler.assign_topics(X_train)
            val_topics = topic_modeler.assign_topics(X_val)
            test_topics = topic_modeler.assign_topics(X_test)
            
            # 提取特征
            keyword_extractor = KeywordExtractor()
            X_train_features = keyword_extractor.extract_features(X_train, train_topics)
            X_val_features = keyword_extractor.extract_features(X_val, val_topics)
            X_test_features = keyword_extractor.extract_features(X_test, test_topics)
            
            # 训练分类器
            classifier = Classifier()
            classifier.train(X_train_features, y_train)
            
            # 在验证集上评估
            y_val_pred = classifier.predict(X_val_features)
            accuracy = accuracy_score(y_val, y_val_pred)
            f1 = f1_score(y_val, y_val_pred, average='macro')
            
            # 获取LDA指标
            perplexity = topic_modeler.get_perplexity()
            topic_clarity = topic_modeler.get_topic_clarity()
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'perplexity': perplexity,
                'topic_clarity': topic_clarity
            }
            
            logger.info(f"参数组合性能: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估参数组合时出错: {e}")
            return {
                'accuracy': 0,
                'f1_score': 0,
                'perplexity': float('inf'),
                'topic_clarity': 0
            }
    
    def optimize(self):
        """
        执行参数调优
        
        Returns:
            dict: 最佳参数组合
        """
        logger.info("开始预处理参数调优...")
        logger.info(f"共评估 {len(self.param_combinations)} 个参数组合")
        
        results = []
        
        # 评估每个参数组合
        for i, params in enumerate(self.param_combinations):
            logger.info(f"评估参数组合 {i+1}/{len(self.param_combinations)}")
            
            metrics = self.evaluate_params(params)
            results.append({
                'params': params,
                'metrics': metrics
            })
            
            # 更新最佳参数
            if (metrics['accuracy'] > self.best_metrics['accuracy'] or
                (metrics['accuracy'] == self.best_metrics['accuracy'] and 
                 metrics['f1_score'] > self.best_metrics['f1_score']) or
                (metrics['accuracy'] == self.best_metrics['accuracy'] and 
                 metrics['f1_score'] == self.best_metrics['f1_score'] and 
                 metrics['perplexity'] < self.best_metrics['perplexity'])):
                
                self.best_metrics = metrics
                self.best_params = params
                logger.info(f"找到新的最佳参数: {params}")
                logger.info(f"最佳性能: {metrics}")
        
        # 保存结果
        self._save_results(results)
        
        logger.info(f"参数调优完成!")
        logger.info(f"最佳参数组合: {self.best_params}")
        logger.info(f"最佳性能指标: {self.best_metrics}")
        
        return self.best_params
    
    def _save_results(self, results):
        """
        保存调优结果
        
        Args:
            results: 调优结果
        """
        # 确保logs目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/preprocessing_optimization_{timestamp}.json'
        
        # 保存结果
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"调优结果已保存到: {filename}")

if __name__ == "__main__":
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='预处理参数调优')
    parser.add_argument('--data', type=str, default='../data/processed_data.csv',
                        help='数据文件路径')
    
    args = parser.parse_args()
    
    # 初始化调优器
    optimizer = PreprocessingOptimizer(args.data)
    
    # 执行调优
    best_params = optimizer.optimize()
    
    # 打印结果
    print("最佳参数组合:")
    print(best_params)
    print("最佳性能指标:")
    print(optimizer.best_metrics)