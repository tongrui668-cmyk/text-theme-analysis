#!/usr/bin/env python3
# 模型监控脚本

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_monitor import ModelMonitor
from src.config import config

def main():
    """主函数"""
    print("=== 模型监控状态检查 ===")
    
    # 初始化监控器
    monitor = ModelMonitor()
    
    # 检查模型文件
    model_path = config.DATA_PATHS['models'] / 'lda_model.pkl'
    
    if not model_path.exists():
        print("❌ 模型文件不存在")
        return
    
    print(f"模型文件: {model_path}")
    print(f"文件大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    # 运行监控
    print("正在生成性能报告...")
    try:
        report = monitor.monitor_model_performance(model_path)
        
        # 输出监控结果
        print("=== 监控结果 ===")
        print(f"时间戳: {report['timestamp']}")
        print()
        
        # 性能指标
        metrics = report['performance_metrics']
        print("性能指标:")
        print(f"  困惑度: {metrics.get('perplexity', 'N/A'):.2f}")
        print(f"  一致性: {metrics.get('coherence', 'N/A'):.3f}")
        print()
        
        # 退化检测
        degradation = metrics.get('degradation', {})
        status = "正常" if not degradation.get('detected', False) else "异常"
        print(f"性能状态: {'✅ 正常' if status == '正常' else '⚠️ 异常'}")
        
        if degradation.get('detected', False):
            print("退化原因:")
            for reason in degradation.get('reasons', []):
                print(f"  - {reason}")
        print()
        
        # 建议
        print("建议:")
        for recommendation in report.get('recommendations', []):
            print(f"  - {recommendation}")
        print()
        
        # 生成警报
        alert = monitor.generate_alert(report)
        if alert:
            print("=== 警报信息 ===")
            print(alert)
        
        print("=== 监控完成 ===")
        
    except Exception as e:
        print(f"❌ 监控失败: {str(e)}")

if __name__ == "__main__":
    main()