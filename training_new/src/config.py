# 配置管理模块
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Config:
    """训练配置类"""
    
    # 数据配置
    DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / '评论和正文.xlsx'
    TEXT_COLUMN = '评论内容'
    
    # 数据划分比例
    TRAIN_SIZE = 0.6
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.2
    
    # 向量化器配置
    COUNT_VECTORIZER = {
        'max_features': 6000,
        'min_df': 3,
        'max_df': 0.85,
        'ngram_range': (1, 2)
    }
    
    TFIDF_VECTORIZER = {
        'max_features': 6000,
        'min_df': 3,
        'max_df': 0.85,
        'ngram_range': (1, 2)
    }
    
    # LDA配置
    LDA = {
        'random_state': 42,
        'max_iter': 500,
        'learning_method': 'online',
        'learning_offset': 15.0,
        'doc_topic_prior': 0.3,
        'topic_word_prior': 0.0001
    }
    
    # 主题数评估范围
    TOPIC_NUMBERS = [4, 5, 6, 7]
    
    # 随机森林分类器配置
    RANDOM_FOREST = {
        'n_estimators': 100,
        'max_depth': 50,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 关键词提取配置
    KEYWORDS = {
        'lda_top_words': 15,
        'tfidf_top_words': 10,
        'combined_top_words': 10
    }
    
    # 模型保存路径
    MODEL_DIR = PROJECT_ROOT / 'data' / 'models'
    
    # 日志保存路径
    LOG_DIR = PROJECT_ROOT / 'training_new' / 'logs'
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 项目根目录
    PROJECT_ROOT = PROJECT_ROOT
    
    # 数据路径配置
    DATA_PATHS = {
        'raw_data': PROJECT_ROOT / 'data' / 'raw',
        'models': PROJECT_ROOT / 'data' / 'models',
        'stopwords': PROJECT_ROOT / 'data' / 'raw' / 'chinese_stopwords.txt',
        'custom_dict': PROJECT_ROOT / 'data' / 'raw' / 'custom_dict.txt',
        'training_data': PROJECT_ROOT / 'data' / 'raw' / '评论和正文.xlsx',
    }
    
    # 模型文件路径
    MODEL_FILES = {
        'lda_model': PROJECT_ROOT / 'data' / 'models' / 'lda_model.pkl',
        'theme_classifier': PROJECT_ROOT / 'data' / 'models' / 'theme_classification_model.pkl',
        'vectorizer': PROJECT_ROOT / 'data' / 'models' / 'vectorizer.pkl',
        'count_vectorizer': PROJECT_ROOT / 'data' / 'models' / 'count_vectorizer.pkl',
        'theme_keywords': PROJECT_ROOT / 'data' / 'models' / 'theme_keywords.pkl',
    }
    
    # 日志配置
    LOG_CONFIG = {
        'log_file': PROJECT_ROOT / 'logs' / 'app.log',
        'log_level': 'INFO',
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
    }
    
    # 文本预处理配置
    PREPROCESSING_CONFIG = {
        'min_text_length': 5,
        'max_text_length': 10000,
        'remove_punctuation': False,  # 中文标点符号处理
        'remove_numbers': False,
        'to_lowercase': False,  # 中文不需要转小写
        'segmenter': 'thulac',  # 'thulac' or 'jieba'
    }
    
    # 主题分类配置
    THEME_CONFIG = {
        'default_themes': [
            '生活方式', '美食', '旅行', '时尚', '美妆', 
            '健身', '科技', '教育', '娱乐', '职场'
        ],
        'keywords_per_theme': 10,
    }

# 创建全局配置实例
config = Config()
