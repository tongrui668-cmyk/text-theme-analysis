"""
配置模块
提供应用配置管理
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    """基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = PROJECT_ROOT / 'data' / 'raw' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # 模型配置
    MODEL_DIR = PROJECT_ROOT / 'data' / 'models'
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = PROJECT_ROOT / 'logs' / 'app.log'
    
    # 调试配置
    DEBUG = False
    
    # 数据路径配置
    DATA_PATHS = {
        'raw_data': PROJECT_ROOT / 'data' / 'raw',
        'models': PROJECT_ROOT / 'data' / 'models',
        'stopwords': PROJECT_ROOT / 'data' / 'processed' / 'stopwords.txt',
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
    
    # 模型参数
    MODEL_CONFIG = {
        'lda': {
            'max_topics': 20,
            'min_topics': 2,
            'random_state': 42,
            'learning_method': 'online',
            'learning_offset': 50.0,
            'max_iter': 1000,
            'batch_size': 128,
            'evaluate_every': -1,
            'verbose': 1,
        },
        'random_forest': {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        'vectorizer': {
            'max_features': 5000,
            'min_df': 1,
            'max_df': 0.95,
            'ngram_range': (1, 2),
        }
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

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    LOG_CONFIG = Config.LOG_CONFIG.copy()
    LOG_CONFIG['log_level'] = 'DEBUG'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    LOG_CONFIG = Config.LOG_CONFIG.copy()
    LOG_CONFIG['log_level'] = 'WARNING'

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    LOG_CONFIG = Config.LOG_CONFIG.copy()
    LOG_CONFIG['log_level'] = 'DEBUG'

# 配置字典
config = {
    'default': DevelopmentConfig,
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}