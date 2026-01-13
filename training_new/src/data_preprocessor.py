# 数据预处理模块

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import config


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, text_preprocessor):
        """
        初始化数据预处理器
        
        Args:
            text_preprocessor: 文本预处理器实例
        """
        self.text_preprocessor = text_preprocessor
    
    def load_data(self):
        """
        加载数据文件
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        print(f"加载数据文件: {config.DATA_FILE}")
        df = pd.read_excel(config.DATA_FILE)
        
        # 检查是否包含 '评论内容' 列
        if config.TEXT_COLUMN not in df.columns:
            raise ValueError(f"数据文件中缺少 '{config.TEXT_COLUMN}' 列，请检查文件。")
        
        print(f"数据加载完成，共 {len(df)} 条记录")
        return df
    
    def preprocess_text(self, text):
        """
        预处理单条文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if pd.isnull(text):
            return ""
        # 使用外部文本预处理器
        processed_words = self.text_preprocessor.preprocess(text)
        return " ".join(processed_words)
    
    def preprocess_data(self, df):
        """
        预处理数据集
        
        Args:
            df: 原始数据
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        print("开始文本预处理...")
        
        # 应用文本预处理
        df['cleaned_text'] = df[config.TEXT_COLUMN].apply(self.preprocess_text)
        
        # 过滤掉空的文本
        df = df[df['cleaned_text'].str.len() > 0]
        print(f"预处理完成，过滤后剩余 {len(df)} 条有效记录")
        
        return df
    
    def split_data(self, df):
        """
        划分数据集
        
        Args:
            df: 预处理后的数据
            
        Returns:
            tuple: (训练集, 验证集, 测试集)
        """
        print("开始数据划分...")
        
        # 第一步：将数据分为训练集（80%）和测试集（20%）
        df_train_val, df_test = train_test_split(
            df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
        )
        
        # 第二步：在训练验证集中再分为训练集（75%）和验证集（25%）
        df_train, df_val = train_test_split(
            df_train_val, test_size=config.VALIDATION_SIZE/(config.TRAIN_SIZE + config.VALIDATION_SIZE), 
            random_state=config.RANDOM_SEED
        )
        
        print(f"数据划分完成:")
        print(f"- 训练集: {len(df_train)} 条记录")
        print(f"- 验证集: {len(df_val)} 条记录")
        print(f"- 测试集: {len(df_test)} 条记录")
        
        return df_train, df_val, df_test
