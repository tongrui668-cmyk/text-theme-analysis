# 分类器模块

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.config import config


class ThemeClassifier:
    """主题分类器"""
    
    def __init__(self):
        """初始化分类器"""
        self.classifier = None
    
    def create_classifier(self):
        """
        创建随机森林分类器
        
        Returns:
            RandomForestClassifier: 配置好的分类器
        """
        classifier = RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST['n_estimators'],
            max_depth=config.RANDOM_FOREST['max_depth'],
            min_samples_split=config.RANDOM_FOREST['min_samples_split'],
            min_samples_leaf=config.RANDOM_FOREST['min_samples_leaf'],
            random_state=config.RANDOM_FOREST['random_state'],
            n_jobs=config.RANDOM_FOREST['n_jobs']
        )
        self.classifier = classifier
        return classifier
    
    def train(self, X_train, y_train):
        """
        训练分类器
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            RandomForestClassifier: 训练好的分类器
        """
        print("开始训练主题分类器...")
        self.classifier.fit(X_train, y_train)
        return self.classifier
    
    def evaluate(self, X_test, y_test, business_labels):
        """
        评估分类器性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            business_labels: 业务标签字典
            
        Returns:
            dict: 评估结果
        """
        print("评估模型性能...")
        
        # 预测
        y_pred = self.classifier.predict(X_test)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 打印详细评估报告
        print("\n分类报告（测试集）:")
        target_names = [business_labels.get(i, f"主题 {i+1}") for i in range(len(business_labels))]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 提取关键指标
        evaluation_results = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'target_names': target_names
        }
        
        # 计算宏平均和加权平均指标
        evaluation_results['macro_avg'] = report['macro avg']
        evaluation_results['weighted_avg'] = report['weighted avg']
        
        return evaluation_results
    
    def predict(self, X):
        """
        预测新数据
        
        Args:
            X: 新数据特征
            
        Returns:
            np.ndarray: 预测结果
        """
        if not self.classifier:
            raise ValueError("分类器未训练")
        return self.classifier.predict(X)
