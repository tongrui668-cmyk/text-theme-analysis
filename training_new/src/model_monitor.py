# 模型性能自动监控模块

import json
import os
from datetime import datetime
from pathlib import Path

from src.config import config
from src.topic_modeler import TopicModeler

class ModelMonitor:
    """模型性能自动监控器"""
    
    def __init__(self):
        """初始化模型监控器"""
        self.monitoring_history = {}
        self.performance_thresholds = {
            'perplexity_increase': 10.0,  # 困惑度增加超过10%视为退化
            'coherence_decrease': 0.1,     # 一致性下降超过0.1视为退化
            'topic_drift': 0.3             # 主题漂移超过30%视为退化
        }
    
    def monitor_model_performance(self, model_path, new_data=None):
        """
        监控模型性能
        
        Args:
            model_path: 模型路径
            new_data: 新数据（可选）
            
        Returns:
            dict: 性能监控结果
        """
        # 加载模型
        modeler = TopicModeler()
        modeler.load_model(model_path)
        
        # 评估模型性能
        performance = {}
        
        # 1. 计算困惑度
        if new_data:
            performance['perplexity'] = modeler.evaluate_perplexity(new_data)
        else:
            performance['perplexity'] = modeler.evaluate_perplexity()
        
        # 2. 计算主题一致性
        performance['coherence'] = modeler.evaluate_coherence()
        
        # 3. 分析主题分布
        performance['topic_distribution'] = modeler.get_topic_distribution()
        
        # 4. 检测性能退化
        performance['degradation'] = self._detect_degradation(performance)
        
        # 5. 生成监控报告
        report = self._generate_performance_report(performance)
        
        # 保存监控历史
        self._save_monitoring_history(model_path, performance)
        
        return report
    
    def _detect_degradation(self, current_performance):
        """
        检测模型性能退化
        
        Args:
            current_performance: 当前性能指标
            
        Returns:
            dict: 退化检测结果
        """
        degradation = {
            'detected': False,
            'reasons': []
        }
        
        # 比较历史性能
        if self.monitoring_history:
            latest_performance = list(self.monitoring_history.values())[-1]
            
            # 检查困惑度增加
            if 'perplexity' in current_performance and 'perplexity' in latest_performance:
                perplexity_increase = (current_performance['perplexity'] - latest_performance['perplexity']) / latest_performance['perplexity'] * 100
                if perplexity_increase > self.performance_thresholds['perplexity_increase']:
                    degradation['detected'] = True
                    degradation['reasons'].append(f"困惑度增加了{perplexity_increase:.2f}%")
            
            # 检查一致性下降
            if 'coherence' in current_performance and 'coherence' in latest_performance:
                coherence_decrease = latest_performance['coherence'] - current_performance['coherence']
                if coherence_decrease > self.performance_thresholds['coherence_decrease']:
                    degradation['detected'] = True
                    degradation['reasons'].append(f"主题一致性下降了{coherence_decrease:.3f}")
        
        return degradation
    
    def _generate_performance_report(self, performance):
        """
        生成性能报告
        
        Args:
            performance: 性能指标
            
        Returns:
            dict: 性能报告
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance,
            'recommendations': []
        }
        
        # 生成建议
        if performance.get('degradation', {}).get('detected', False):
            report['recommendations'].append("模型性能出现退化，建议重新训练")
        else:
            report['recommendations'].append("模型性能稳定，无需立即更新")
        
        # 基于具体指标的建议
        if 'perplexity' in performance and performance['perplexity'] > 1000:
            report['recommendations'].append("困惑度较高，建议调整模型参数")
        
        if 'coherence' in performance and performance['coherence'] < 0.4:
            report['recommendations'].append("主题一致性较低，建议优化主题数量")
        
        return report
    
    def _save_monitoring_history(self, model_path, performance):
        """
        保存监控历史
        
        Args:
            model_path: 模型路径
            performance: 性能指标
        """
        timestamp = datetime.now().isoformat()
        self.monitoring_history[timestamp] = performance
        
        # 保存到文件
        monitor_dir = Path(model_path).parent / 'monitoring'
        monitor_dir.mkdir(exist_ok=True)
        
        history_file = monitor_dir / 'performance_history.json'
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                existing_history = json.load(f)
        else:
            existing_history = {}
        
        existing_history.update(self.monitoring_history)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, ensure_ascii=False, indent=2)
    
    def generate_alert(self, report):
        """
        生成性能警报
        
        Args:
            report: 性能报告
            
        Returns:
            str: 警报信息
        """
        if report['performance_metrics'].get('degradation', {}).get('detected', False):
            alert = f"⚠️ 模型性能警报 ({report['timestamp']})\n"
            alert += "检测到性能退化：\n"
            for reason in report['performance_metrics']['degradation']['reasons']:
                alert += f"- {reason}\n"
            alert += "建议：\n"
            for recommendation in report['recommendations']:
                alert += f"- {recommendation}\n"
            return alert
        return None
    
    def compare_models(self, model_path1, model_path2):
        """
        比较两个模型的性能
        
        Args:
            model_path1: 第一个模型路径
            model_path2: 第二个模型路径
            
        Returns:
            dict: 模型比较结果
        """
        # 监控两个模型
        report1 = self.monitor_model_performance(model_path1)
        report2 = self.monitor_model_performance(model_path2)
        
        comparison = {
            'model1': model_path1,
            'model2': model_path2,
            'performance_metrics': {
                'model1': report1['performance_metrics'],
                'model2': report2['performance_metrics']
            },
            'better_model': None
        }
        
        # 确定更好的模型
        score1 = self._calculate_model_score(report1['performance_metrics'])
        score2 = self._calculate_model_score(report2['performance_metrics'])
        
        if score1 > score2:
            comparison['better_model'] = model_path1
        else:
            comparison['better_model'] = model_path2
        
        return comparison
    
    def _calculate_model_score(self, performance):
        """
        计算模型综合得分
        
        Args:
            performance: 性能指标
            
        Returns:
            float: 模型得分
        """
        score = 0
        
        # 困惑度（越低越好）
        if 'perplexity' in performance:
            try:
                # 添加保护，避免除零错误
                perplexity = max(1, performance['perplexity'])  # 确保perplexity至少为1
                perplexity_score = max(0, 100 - (perplexity / 100))
                score += perplexity_score * 0.4
            except Exception as e:
                print(f"计算困惑度得分时出错: {str(e)}")
                # 使用默认得分
                score += 50 * 0.4
        
        # 一致性（越高越好）
        if 'coherence' in performance:
            try:
                coherence_score = performance['coherence'] * 200
                score += coherence_score * 0.6
            except Exception as e:
                print(f"计算一致性得分时出错: {str(e)}")
                # 使用默认得分
                score += 50 * 0.6
        
        return score

if __name__ == "__main__":
    # 示例用法
    monitor = ModelMonitor()
    
    # 监控当前模型
    model_path = config.DATA_PATHS['models'] / 'lda_model.pkl'
    if model_path.exists():
        report = monitor.monitor_model_performance(model_path)
        
        # 生成警报
        alert = monitor.generate_alert(report)
        if alert:
            print(alert)
        
        # 保存报告
        report_file = config.DATA_PATHS['models'] / 'monitoring' / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"性能报告已保存至: {report_file}")
    else:
        print("未找到模型文件")
