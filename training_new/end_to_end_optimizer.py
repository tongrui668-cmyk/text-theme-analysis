#!/usr/bin/env python3
"""
端到端优化框架
将预处理、特征提取、模型训练集成到同一优化流程
"""

import os
import sys
import logging
import json
from datetime import datetime
from sklearn.model_selection import ParameterGrid
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
        logging.FileHandler('logs/end_to_end_optimization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EndToEndOptimizer:
    """端到端优化器"""
    
    def __init__(self, data_file):
        """
        初始化端到端优化器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.best_config = None
        self.best_metrics = {
            'accuracy': 0,
            'f1_score': 0,
            'perplexity': float('inf'),
            'topic_clarity': 0
        }
        self.config_grid = self._generate_config_grid()
    
    def _generate_config_grid(self):
        """
        生成配置网格
        
        Returns:
            ParameterGrid: 配置网格
        """
        # 定义参数空间
        param_grid = {
            # 预处理参数
            'preprocessing': {
                'semantic_strategy': ['full', 'moderate', 'minimal'],
                'stopword_strategy': ['full', 'moderate', 'minimal'],
                'noise_filter_strategy': ['full', 'moderate', 'minimal']
            },
            
            # 特征提取参数
            'feature_extraction': {
                'use_lda': [True],
                'use_tfidf': [True],
                'topic_numbers': [4, 5, 6, 7, 8],
                'tfidf_max_features': [4000, 6000, 8000]
            },
            
            # 分类器参数
            'classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        return ParameterGrid(param_grid)
    
    def _get_preprocessor_with_config(self, preprocessing_config):
        """
        根据配置获取预处理器
        
        Args:
            preprocessing_config: 预处理配置
            
        Returns:
            EnhancedTextPreprocessor: 配置好的预处理器
        """
        # 创建预处理器
        preprocessor = EnhancedTextPreprocessor()
        
        # 根据配置调整预处理器
        # 这里可以根据需要修改预处理器的行为
        
        return preprocessor
    
    def evaluate_config(self, config):
        """
        评估特定配置的性能
        
        Args:
            config: 完整配置
            
        Returns:
            dict: 性能指标
        """
        try:
            logger.info(f"评估配置: {config}")
            
            # 提取配置
            preprocessing_config = config['preprocessing']
            feature_config = config['feature_extraction']
            classifier_config = config['classifier']
            
            # 获取配置好的预处理器
            preprocessor = self._get_preprocessor_with_config(preprocessing_config)
            
            # 初始化数据预处理器
            data_preprocessor = DataPreprocessor(preprocessor)
            
            # 加载和预处理数据
            data_preprocessor.load_data(self.data_file)
            data_preprocessor.preprocess_data()
            
            # 划分数据
            X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.split_data()
            
            # 初始化主题模型
            topic_modeler = TopicModeler()
            
            # 训练LDA模型
            num_topics = feature_config['topic_numbers']
            topic_modeler.train_final_model(X_train, num_topics)
            
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
            
            # 初始化分类器
            classifier = Classifier()
            
            # 训练分类器
            classifier.train(X_train_features, y_train, **classifier_config)
            
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
            
            logger.info(f"配置性能: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估配置时出错: {e}")
            return {
                'accuracy': 0,
                'f1_score': 0,
                'perplexity': float('inf'),
                'topic_clarity': 0
            }
    
    def optimize(self):
        """
        执行端到端优化
        
        Returns:
            dict: 最佳配置
        """
        logger.info("开始端到端优化...")
        logger.info(f"共评估 {len(self.config_grid)} 个配置")
        
        results = []
        
        # 评估每个配置
        for i, config in enumerate(self.config_grid):
            logger.info(f"评估配置 {i+1}/{len(self.config_grid)}")
            
            metrics = self.evaluate_config(config)
            results.append({
                'config': config,
                'metrics': metrics
            })
            
            # 更新最佳配置
            if (metrics['accuracy'] > self.best_metrics['accuracy'] or
                (metrics['accuracy'] == self.best_metrics['accuracy'] and 
                 metrics['f1_score'] > self.best_metrics['f1_score']) or
                (metrics['accuracy'] == self.best_metrics['accuracy'] and 
                 metrics['f1_score'] == self.best_metrics['f1_score'] and 
                 metrics['perplexity'] < self.best_metrics['perplexity'])):
                
                self.best_metrics = metrics
                self.best_config = config
                logger.info(f"找到新的最佳配置: {config}")
                logger.info(f"最佳性能: {metrics}")
        
        # 保存结果
        self._save_results(results)
        
        logger.info(f"端到端优化完成!")
        logger.info(f"最佳配置: {self.best_config}")
        logger.info(f"最佳性能指标: {self.best_metrics}")
        
        return self.best_config
    
    def _save_results(self, results):
        """
        保存优化结果
        
        Args:
            results: 优化结果
        """
        # 确保logs目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/end_to_end_optimization_{timestamp}.json'
        
        # 保存结果
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化结果已保存到: {filename}")
    
    def train_best_model(self, best_config):
        """
        使用最佳配置训练最终模型
        
        Args:
            best_config: 最佳配置
            
        Returns:
            tuple: 训练好的模型组件
        """
        try:
            logger.info(f"使用最佳配置训练最终模型: {best_config}")
            
            # 提取配置
            preprocessing_config = best_config['preprocessing']
            feature_config = best_config['feature_extraction']
            classifier_config = best_config['classifier']
            
            # 获取配置好的预处理器
            preprocessor = self._get_preprocessor_with_config(preprocessing_config)
            
            # 初始化数据预处理器
            data_preprocessor = DataPreprocessor(preprocessor)
            
            # 加载和预处理数据
            data_preprocessor.load_data(self.data_file)
            data_preprocessor.preprocess_data()
            
            # 划分数据
            X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessor.split_data()
            
            # 合并训练集和验证集用于最终训练
            X_train_final = X_train + X_val
            y_train_final = y_train + y_val
            
            # 初始化主题模型
            topic_modeler = TopicModeler()
            
            # 训练LDA模型
            num_topics = feature_config['topic_numbers']
            topic_modeler.train_final_model(X_train_final, num_topics)
            
            # 生成业务标签
            label_generator = LabelGenerator(topic_modeler)
            business_labels = label_generator.generate_business_labels()
            
            # 分配主题
            train_topics_final = topic_modeler.assign_topics(X_train_final)
            test_topics = topic_modeler.assign_topics(X_test)
            
            # 提取特征
            keyword_extractor = KeywordExtractor()
            X_train_features = keyword_extractor.extract_features(X_train_final, train_topics_final)
            X_test_features = keyword_extractor.extract_features(X_test, test_topics)
            
            # 初始化分类器
            classifier = Classifier()
            
            # 训练分类器
            classifier.train(X_train_features, y_train_final, **classifier_config)
            
            # 在测试集上评估
            y_test_pred = classifier.predict(X_test_features)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='macro')
            
            logger.info(f"最终模型测试集性能:")
            logger.info(f"准确率: {test_accuracy}")
            logger.info(f"F1分数: {test_f1}")
            
            # 生成报告
            report_generator = ReportGenerator()
            report_generator.generate_report(
                topic_modeler, classifier, 
                X_test, y_test, y_test_pred,
                business_labels
            )
            
            # 保存模型
            model_saver = ModelSaver()
            model_saver.save_all_models(
                topic_modeler, keyword_extractor, classifier,
                business_labels
            )
            
            return {
                'topic_modeler': topic_modeler,
                'keyword_extractor': keyword_extractor,
                'classifier': classifier,
                'business_labels': business_labels,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1
            }
            
        except Exception as e:
            logger.error(f"训练最终模型时出错: {e}")
            raise

if __name__ == "__main__":
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='端到端优化')
    parser.add_argument('--data', type=str, default='../data/processed_data.csv',
                        help='数据文件路径')
    parser.add_argument('--optimize', action='store_true',
                        help='是否执行优化')
    parser.add_argument('--train', action='store_true',
                        help='是否训练最终模型')
    
    args = parser.parse_args()
    
    # 初始化优化器
    optimizer = EndToEndOptimizer(args.data)
    
    if args.optimize:
        # 执行优化
        best_config = optimizer.optimize()
        
        # 打印结果
        print("最佳配置:")
        print(best_config)
        print("最佳性能指标:")
        print(optimizer.best_metrics)
        
    if args.train:
        # 加载最佳配置
        import glob
        import json
        
        # 查找最新的优化结果
        optimization_files = glob.glob('logs/end_to_end_optimization_*.json')
        if optimization_files:
            latest_file = max(optimization_files, key=os.path.getmtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 找到最佳配置
            best_result = max(results, key=lambda x: x['metrics']['accuracy'])
            best_config = best_result['config']
            
            print(f"使用从 {latest_file} 加载的最佳配置")
        else:
            # 如果没有优化结果，使用默认配置
            best_config = {
                'preprocessing': {
                    'semantic_strategy': 'full',
                    'stopword_strategy': 'full',
                    'noise_filter_strategy': 'full'
                },
                'feature_extraction': {
                    'use_lda': True,
                    'use_tfidf': True,
                    'topic_numbers': 7,
                    'tfidf_max_features': 6000
                },
                'classifier': {
                    'n_estimators': 200,
                    'max_depth': None,
                    'min_samples_split': 2
                }
            }
            
            print("使用默认最佳配置")
        
        # 训练最终模型
        model_results = optimizer.train_best_model(best_config)
        
        # 打印结果
        print("最终模型性能:")
        print(f"测试集准确率: {model_results['test_accuracy']}")
        print(f"测试集F1分数: {model_results['test_f1']}")