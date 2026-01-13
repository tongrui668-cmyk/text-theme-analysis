#!/usr/bin/env python3
# 自动部署脚本

import os
import sys
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.topic_modeler import TopicModeler
from src.data_preprocessor import DataPreprocessor
from src.topic_name_optimizer import TopicNameOptimizer
from src.model_monitor import ModelMonitor

class ModelDeployer:
    """模型自动部署器"""
    
    def __init__(self):
        """初始化模型部署器"""
        self.deployment_history = []
        self.deployment_dir = config.DATA_PATHS['models'] / 'deployments'
        self.deployment_dir.mkdir(exist_ok=True)
    
    def deploy_model(self, new_data=None, config_updates=None):
        """
        部署模型
        
        Args:
            new_data: 新数据（可选）
            config_updates: 配置更新（可选）
            
        Returns:
            dict: 部署结果
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deployment_path = self.deployment_dir / deployment_id
        deployment_path.mkdir(exist_ok=True)
        
        # 1. 准备数据
        print("1. 准备数据...")
        from src.text_preprocessor import TextPreprocessor
        text_processor = TextPreprocessor()
        processor = DataPreprocessor(text_processor)
        
        if new_data:
            data = text_processor.preprocess(new_data)
        else:
            # 使用默认训练数据
            df = processor.load_data()
            df = processor.preprocess_data(df)
            data = df['cleaned_text'].tolist()
        
        # 2. 训练新模型
        print("2. 训练新模型...")
        modeler = TopicModeler()
        
        if config_updates:
            # 这里可以添加配置更新逻辑
            pass
        
        # 评估最佳主题数
        best_n_topics = modeler.evaluate_topic_numbers(data)
        
        # 训练最终模型
        modeler.train_final_model(data, best_n_topics)
        
        # 保存模型
        modeler.save_model()
        
        # 3. 优化主题名称
        print("3. 优化主题名称...")
        optimizer = TopicNameOptimizer()
        topics = modeler.get_topics()
        
        optimized_topics = {}
        for topic_idx, keywords in topics.items():
            current_name = f"主题{topic_idx}"
            optimized_name = optimizer.optimize_topic_name(current_name, keywords)
            optimized_topics[optimized_name] = keywords
        
        # 4. 评估新模型
        print("4. 评估新模型...")
        monitor = ModelMonitor()
        new_model_performance = monitor.monitor_model_performance(modeler.model_path)
        
        # 5. 与现有模型比较
        print("5. 与现有模型比较...")
        current_model_path = config.DATA_PATHS['models'] / 'lda_model.pkl'
        deployment_decision = {
            'deploy_new': False,
            'reason': ''
        }
        
        if current_model_path.exists():
            comparison = monitor.compare_models(current_model_path, modeler.model_path)
            if comparison['better_model'] == str(modeler.model_path):
                deployment_decision['deploy_new'] = True
                deployment_decision['reason'] = "新模型性能优于现有模型"
            else:
                deployment_decision['deploy_new'] = False
                deployment_decision['reason'] = "现有模型性能更优"
        else:
            deployment_decision['deploy_new'] = True
            deployment_decision['reason'] = "无现有模型，部署新模型"
        
        # 6. 执行部署
        print("6. 执行部署...")
        if deployment_decision['deploy_new']:
            # 备份现有模型
            if current_model_path.exists():
                backup_path = config.DATA_PATHS['models'] / f'backup_lda_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                shutil.copy2(current_model_path, backup_path)
                print(f"已备份现有模型至: {backup_path}")
            
            # 部署新模型
            shutil.copy2(modeler.model_path, current_model_path)
            print(f"已部署新模型至: {current_model_path}")
            
            # 保存优化后的主题
            optimized_topics_path = config.DATA_PATHS['models'] / 'optimized_topics.json'
            with open(optimized_topics_path, 'w', encoding='utf-8') as f:
                json.dump(optimized_topics, f, ensure_ascii=False, indent=2)
            print(f"已保存优化后的主题至: {optimized_topics_path}")
        
        # 7. 生成部署报告
        print("7. 生成部署报告...")
        deployment_report = {
            'deployment_id': deployment_id,
            'timestamp': datetime.now().isoformat(),
            'deployment_path': str(deployment_path),
            'new_model_path': str(modeler.model_path),
            'current_model_path': str(current_model_path),
            'deployment_decision': deployment_decision,
            'model_performance': new_model_performance,
            'optimized_topics': optimized_topics,
            'config_updates': config_updates
        }
        
        # 保存部署报告
        report_path = deployment_path / 'deployment_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(deployment_report, f, ensure_ascii=False, indent=2)
        
        # 保存部署历史
        self._save_deployment_history(deployment_report)
        
        print(f"部署完成！报告已保存至: {report_path}")
        return deployment_report
    
    def _save_deployment_history(self, deployment_report):
        """
        保存部署历史
        
        Args:
            deployment_report: 部署报告
        """
        self.deployment_history.append(deployment_report)
        
        # 保存到文件
        history_path = self.deployment_dir / 'deployment_history.json'
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                existing_history = json.load(f)
        else:
            existing_history = []
        
        existing_history.append(deployment_report)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, ensure_ascii=False, indent=2)
    
    def rollback_deployment(self, deployment_id=None):
        """
        回滚部署
        
        Args:
            deployment_id: 部署ID（可选，默认回滚到上一次部署）
            
        Returns:
            dict: 回滚结果
        """
        # 加载部署历史
        history_path = self.deployment_dir / 'deployment_history.json'
        if not history_path.exists():
            return {
                'success': False,
                'reason': '无部署历史'
            }
        
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        if not history:
            return {
                'success': False,
                'reason': '部署历史为空'
            }
        
        # 找到目标部署
        if deployment_id:
            target_deployment = next((d for d in history if d['deployment_id'] == deployment_id), None)
        else:
            # 默认回滚到上一次部署
            if len(history) >= 2:
                target_deployment = history[-2]
            else:
                return {
                    'success': False,
                    'reason': '部署历史不足，无法回滚'
                }
        
        if not target_deployment:
            return {
                'success': False,
                'reason': '未找到目标部署'
            }
        
        # 执行回滚
        print(f"回滚到部署: {target_deployment['deployment_id']}")
        
        # 备份当前模型
        current_model_path = config.DATA_PATHS['models'] / 'lda_model.pkl'
        if current_model_path.exists():
            backup_path = config.DATA_PATHS['models'] / f'rollback_backup_lda_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            shutil.copy2(current_model_path, backup_path)
            print(f"已备份当前模型至: {backup_path}")
        
        # 恢复目标模型
        target_model_path = Path(target_deployment['new_model_path'])
        if target_model_path.exists():
            shutil.copy2(target_model_path, current_model_path)
            print(f"已恢复模型至: {current_model_path}")
            
            # 恢复主题名称
            if 'optimized_topics' in target_deployment:
                optimized_topics_path = config.DATA_PATHS['models'] / 'optimized_topics.json'
                with open(optimized_topics_path, 'w', encoding='utf-8') as f:
                    json.dump(target_deployment['optimized_topics'], f, ensure_ascii=False, indent=2)
                print(f"已恢复主题名称至: {optimized_topics_path}")
            
            return {
                'success': True,
                'deployment_id': target_deployment['deployment_id'],
                'timestamp': target_deployment['timestamp']
            }
        else:
            return {
                'success': False,
                'reason': '目标模型文件不存在'
            }
    
    def get_deployment_status(self):
        """
        获取部署状态
        
        Returns:
            dict: 部署状态
        """
        current_model_path = config.DATA_PATHS['models'] / 'lda_model.pkl'
        status = {
            'current_model': None,
            'deployments': [],
            'last_deployment': None
        }
        
        if current_model_path.exists():
            status['current_model'] = {
                'path': str(current_model_path),
                'size': current_model_path.stat().st_size / 1024 / 1024,  # MB
                'modified': datetime.fromtimestamp(current_model_path.stat().st_mtime).isoformat()
            }
        
        # 加载部署历史
        history_path = self.deployment_dir / 'deployment_history.json'
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            status['deployments'] = history
            if history:
                status['last_deployment'] = history[-1]
        
        return status

def main():
    """主函数"""
    deployer = ModelDeployer()
    
    # 获取部署状态
    print("=== 当前部署状态 ===")
    status = deployer.get_deployment_status()
    if status['current_model']:
        print(f"当前模型: {status['current_model']['path']}")
        print(f"模型大小: {status['current_model']['size']:.2f} MB")
        print(f"修改时间: {status['current_model']['modified']}")
    else:
        print("当前无部署的模型")
    
    if status['last_deployment']:
        print(f"最后部署: {status['last_deployment']['deployment_id']}")
        print(f"部署时间: {status['last_deployment']['timestamp']}")
        print(f"部署结果: {'成功' if status['last_deployment']['deployment_decision']['deploy_new'] else '未部署'}")
    
    # 执行部署
    print("\n=== 执行部署 ===")
    try:
        # 可选：传入新数据或配置更新
        # 例如: deployment = deployer.deploy_model(new_data=data, config_updates={'n_components': 10})
        deployment = deployer.deploy_model()
        print(f"\n部署结果: {'成功' if deployment['deployment_decision']['deploy_new'] else '未部署'}")
        print(f"部署原因: {deployment['deployment_decision']['reason']}")
        
        # 打印模型性能
        if 'model_performance' in deployment:
            performance = deployment['model_performance']['performance_metrics']
            print(f"\n新模型性能:")
            if 'perplexity' in performance:
                print(f"困惑度: {performance['perplexity']:.2f}")
            if 'coherence' in performance:
                print(f"一致性: {performance['coherence']:.3f}")
        
    except Exception as e:
        print(f"部署失败: {str(e)}")

if __name__ == "__main__":
    main()
