# 报告生成模块

import os
import pandas as pd
from src.config import config


class ReportGenerator:
    """训练报告生成器"""
    
    def __init__(self):
        """初始化报告生成器"""
        pass
    
    def generate_report(self, training_results):
        """
        生成训练报告
        
        Args:
            training_results: 训练结果字典
            
        Returns:
            str: 报告文件路径
        """
        # 创建日志目录
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        report_path = os.path.join(config.LOG_DIR, 'train_log_new.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # 基本信息
            f.write("=== 模型训练报告 ===\n")
            f.write(f"训练时间: {pd.Timestamp.now()}\n")
            f.write(f"总数据量: {training_results['total_samples']}\n")
            f.write(f"训练集: {training_results['train_size']} 条记录\n")
            f.write(f"验证集: {training_results['val_size']} 条记录\n")
            f.write(f"测试集: {training_results['test_size']} 条记录\n")
            f.write(f"特征维度: {training_results['feature_dim']}\n")
            f.write(f"验证集准确率: {training_results['val_accuracy']:.4f}\n")
            f.write(f"测试集准确率: {training_results['test_accuracy']:.4f}\n")
            f.write(f"LDA困惑度: {training_results['perplexity']:.2f}\n")
            f.write(f"最佳主题数: {training_results['n_topics']}\n")
            f.write(f"主题数评估结果:\n")
            for result in training_results['evaluation_results']:
                if 'topic_coherence' in result:
                    f.write(f"  主题数={result['n_topics']}: 困惑度={result['perplexity']:.2f}, 主题清晰度={result['avg_topic_clarity']:.4f}, 主题一致性={result['topic_coherence']:.4f}\n")
                else:
                    f.write(f"  主题数={result['n_topics']}: 困惑度={result['perplexity']:.2f}, 主题清晰度={result['avg_topic_clarity']:.4f}\n")
            f.write("\n")
            
            # 业务标签
            f.write("=== 主题业务标签 ===\n")
            for topic_id, label in training_results['business_labels'].items():
                f.write(f"{label}\n")
            f.write("\n")
            
            # LDA主题关键词
            f.write("=== LDA主题关键词 ===\n")
            for topic_idx, topic in enumerate(training_results['lda_model'].components_):
                topic_label = training_results['business_labels'].get(topic_idx, f"主题 {topic_idx+1}")
                top_words = [training_results['vectorizer'].get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
                f.write(f"{topic_label}: {', '.join(top_words)}\n")
            
            # TF-IDF主题关键词
            f.write("\n=== TF-IDF主题关键词 ===\n")
            for label, words in training_results['tfidf_keywords'].items():
                f.write(f"{label}: {', '.join(words)}\n")
            
            # 组合主题关键词
            f.write("\n=== 组合主题关键词 ===\n")
            for label, words in training_results['combined_keywords'].items():
                f.write(f"{label}: {', '.join(words)}\n")
        
        print(f"训练报告已生成: {report_path}")
        return report_path
