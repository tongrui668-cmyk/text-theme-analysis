#!/usr/bin/env python3
# LDA参数自动调优脚本（优化版）

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.topic_modeler import TopicModeler
from src.data_preprocessor import DataPreprocessor
from src.text_preprocessor import TextPreprocessor

class LDAParameterTuner:
    """LDA参数自动调优器"""
    
    def __init__(self):
        """初始化参数调优器"""
        # 定义要测试的参数范围（基于初步测试结果，只测试最有希望的组合）
        self.parameter_grid = {
            'max_iter': [500, 800, 1000],
            'doc_topic_prior': [0.1],
            'topic_word_prior': [0.0001, 0.001],
            'learning_offset': [15.0]
        }
        
        # 加载数据
        self.data = self._load_data()
    
    def _load_data(self):
        """加载并预处理数据"""
        print("加载并预处理数据...")
        text_processor = TextPreprocessor()
        processor = DataPreprocessor(text_processor)
        df = processor.load_data()
        df = processor.preprocess_data(df)
        return df['cleaned_text'].tolist()
    
    def tune_parameters(self):
        """
        自动调优LDA参数
        
        Returns:
            dict: 最佳参数组合和对应的性能指标
        """
        print("开始自动调优LDA参数...")
        
        best_params = None
        best_perplexity = float('inf')
        best_coherence = 0
        
        total_combinations = np.prod([len(v) for v in self.parameter_grid.values()])
        print(f"总共需要测试 {total_combinations} 种参数组合...")
        
        # 生成所有参数组合
        from itertools import product
        param_combinations = list(product(
            self.parameter_grid['max_iter'],
            self.parameter_grid['doc_topic_prior'],
            self.parameter_grid['topic_word_prior'],
            self.parameter_grid['learning_offset']
        ))
        
        # 测试每个参数组合
        for i, (max_iter, doc_topic_prior, topic_word_prior, learning_offset) in enumerate(param_combinations):
            print(f"\n测试参数组合 {i+1}/{total_combinations}:")
            print(f"max_iter={max_iter}, doc_topic_prior={doc_topic_prior}, topic_word_prior={topic_word_prior}, learning_offset={learning_offset}")
            
            # 创建并配置TopicModeler
            modeler = TopicModeler()
            
            # 临时修改配置
            original_config = config.LDA.copy()
            config.LDA['max_iter'] = max_iter
            config.LDA['doc_topic_prior'] = doc_topic_prior
            config.LDA['topic_word_prior'] = topic_word_prior
            config.LDA['learning_offset'] = learning_offset
            
            try:
                # 评估最佳主题数
                best_n_topics = modeler.evaluate_topic_numbers(self.data)
                
                # 训练最终模型
                modeler.train_final_model(self.data, best_n_topics)
                
                # 评估性能
                perplexity = modeler.evaluate_perplexity()
                coherence = modeler.evaluate_coherence()
                
                print(f"  困惑度: {perplexity:.2f}, 一致性: {coherence:.3f}, 最佳主题数: {best_n_topics}")
                
                # 更新最佳参数
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_coherence = coherence
                    best_params = {
                        'max_iter': max_iter,
                        'doc_topic_prior': doc_topic_prior,
                        'topic_word_prior': topic_word_prior,
                        'learning_offset': learning_offset,
                        'n_topics': best_n_topics
                    }
                    print(f"  🎯 找到新的最佳参数组合!")
                    
            except Exception as e:
                print(f"  ❌ 测试失败: {str(e)}")
            finally:
                # 恢复原始配置
                config.LDA = original_config
        
        print("\n" + "="*60)
        print("参数调优完成!")
        print(f"最佳参数组合: {best_params}")
        print(f"最低困惑度: {best_perplexity:.2f}")
        print(f"最佳一致性: {best_coherence:.3f}")
        print("="*60)
        
        return {
            'best_params': best_params,
            'best_perplexity': best_perplexity,
            'best_coherence': best_coherence
        }

def main():
    """主函数"""
    tuner = LDAParameterTuner()
    result = tuner.tune_parameters()
    
    # 保存调优结果
    import json
    result_path = Path(__file__).parent.parent / 'tuning_results.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n调优结果已保存至: {result_path}")

if __name__ == "__main__":
    main()
