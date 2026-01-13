#!/usr/bin/env python3
# 基本测试脚本

import os
import sys
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.topic_modeler import TopicModeler
from src.topic_name_optimizer import TopicNameOptimizer
from src.model_monitor import ModelMonitor
from src.data_preprocessor import DataPreprocessor

class TestCoreFunctionality(unittest.TestCase):
    """核心功能测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.model_path = config.MODEL_FILES['lda_model']
    
    def test_config_loading(self):
        """测试配置文件加载"""
        print("测试配置文件加载...")
        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, 'PROJECT_ROOT'))
        self.assertTrue(hasattr(config, 'LDA'))
        self.assertTrue(hasattr(config, 'DATA_PATHS'))
        print("配置文件加载测试通过")
    
    def test_data_processor(self):
        """测试数据处理器"""
        print("测试数据处理器...")
        # 导入TextPreprocessor以正确初始化DataPreprocessor
        from src.text_preprocessor import TextPreprocessor
        text_processor = TextPreprocessor()
        processor = DataPreprocessor(text_processor)
        # 测试数据加载
        data = processor.load_data()
        import pandas as pd
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        print("数据处理器测试通过")
    
    def test_topic_modeler(self):
        """测试主题建模器"""
        print("测试主题建模器...")
        modeler = TopicModeler()
        
        # 测试主题建模器初始化
        self.assertIsInstance(modeler, TopicModeler)
        print("主题建模器测试通过")
    
    def test_topic_name_optimizer(self):
        """测试主题名称优化器"""
        print("测试主题名称优化器...")
        optimizer = TopicNameOptimizer()
        
        # 测试单个主题名称优化
        test_topic = "测试主题"
        test_keywords = ["社交", "互动", "聊天", "朋友"]
        optimized_name = optimizer.optimize_topic_name(test_topic, test_keywords)
        
        self.assertIsInstance(optimized_name, str)
        self.assertGreater(len(optimized_name), 0)
        print(f"主题名称优化测试通过，优化结果: {optimized_name}")
    
    def test_model_monitor(self):
        """测试模型监控器"""
        print("测试模型监控器...")
        monitor = ModelMonitor()
        
        # 测试模型监控器初始化
        self.assertIsInstance(monitor, ModelMonitor)
        print("模型监控器测试通过")

class TestDeployment(unittest.TestCase):
    """部署功能测试"""
    
    def test_deployment_readiness(self):
        """测试部署就绪性"""
        print("测试部署就绪性...")
        
        # 检查必要的目录是否存在
        required_dirs = [
            config.DATA_PATHS['raw_data'],
            config.DATA_PATHS['models'],
            Path(__file__).parent.parent / 'scripts'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(dir_path.exists(), f"目录不存在: {dir_path}")
        
        # 检查必要的文件是否存在
        required_files = [
            config.DATA_FILE,
            config.DATA_PATHS['stopwords'],
            Path(__file__).parent.parent / 'scripts' / 'deploy_model.py'
        ]
        
        for file_path in required_files:
            if file_path != config.DATA_PATHS['stopwords']:  # 停用词文件可选
                self.assertTrue(file_path.exists(), f"文件不存在: {file_path}")
        
        print("部署就绪性测试通过")

if __name__ == '__main__':
    print("开始运行测试...")
    unittest.main(verbosity=2)
