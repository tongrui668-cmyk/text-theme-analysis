"""
文本数据预处理器
提供统一的文本清洗、分词和预处理功能
"""
import re
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path

from src.config import config
from src.logger import Logger, log_function_call, log_error, log_info

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, segmenter: str = None):
        """
        初始化预处理器
        
        Args:
            segmenter: 分词器类型，'thulac' 或 'jieba'，默认使用配置文件中的设置
        """
        self.segmenter = segmenter or config['default'].PREPROCESSING_CONFIG['segmenter']
        self.logger = Logger.get_logger(__name__)
        self.stopwords = self._load_stopwords()
        self.thulac_model = None
        self.thulac = None  # thulac模块引用
        self.jieba = None  # jieba模块引用
        
        # 初始化分词器
        self._init_segmenter()
        
        log_info(f"文本预处理器初始化完成，使用分词器: {self.segmenter}")
    
    @log_function_call
    def _init_segmenter(self):
        """初始化分词器"""
        try:
            if self.segmenter == 'thulac':
                # 动态导入thulac
                try:
                    import thulac
                    self.thulac = thulac
                    self.thulac_model = thulac.thulac(seg_only=True)
                    self.logger.info("THULAC分词器初始化成功")
                except ImportError:
                    self.logger.warning("THULAC模块未安装，切换到jieba分词器")
                    self.segmenter = 'jieba'
                    
                    # 加载自定义词典（如果存在）
                    self._import_jieba()
                    custom_dict_path = config['default'].DATA_PATHS.get('custom_dict')
                    if custom_dict_path and custom_dict_path.exists():
                        self.jieba.load_userdict(str(custom_dict_path))
                        self.logger.info(f"加载自定义词典: {custom_dict_path}")
                    self.logger.info("jieba分词器初始化成功")
            elif self.segmenter == 'jieba':
                # 加载自定义词典（如果存在）
                self._import_jieba()
                custom_dict_path = config['default'].DATA_PATHS.get('custom_dict')
                if custom_dict_path and custom_dict_path.exists():
                    self.jieba.load_userdict(str(custom_dict_path))
                    self.logger.info(f"加载自定义词典: {custom_dict_path}")
                self.logger.info("jieba分词器初始化成功")
            else:
                raise ValueError(f"不支持的分词器: {self.segmenter}")
        except Exception as e:
            log_error(e, "分词器初始化失败")
            # 回退到jieba
            self.segmenter = 'jieba'
            self.logger.warning("回退到jieba分词器")
    
    def _import_jieba(self):
        """延迟导入jieba模块"""
        try:
            if self.jieba is None:
                import jieba
                self.jieba = jieba
        except ImportError as e:
            self.logger.warning(f"jieba模块导入失败: {e}")
            self.jieba = None
    
    @log_function_call
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        stopwords = set()
        stopwords_files = [
            config['default'].DATA_PATHS.get('stopwords'),
            Path(__file__).parent.parent / 'data' / 'raw' / '停用词表.txt'
        ]
        
        for file_path in stopwords_files:
            if file_path and file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word:
                                stopwords.add(word)
                    self.logger.info(f"加载停用词表: {file_path}, 词数: {len(stopwords)}")
                except Exception as e:
                    log_error(e, f"加载停用词表失败: {file_path}")
        
        # 添加一些默认停用词
        default_stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        stopwords.update(default_stopwords)
        
        return stopwords
    
    @log_function_call
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 转换为小写（如果配置要求）
        if config['default'].PREPROCESSING_CONFIG.get('to_lowercase', False):
            text = text.lower()
        
        # 去除数字（如果配置要求）
        if config['default'].PREPROCESSING_CONFIG.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        # 去除标点符号（如果配置要求）- 只去除英文标点，保留中文
        if config['default'].PREPROCESSING_CONFIG.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
        
        return text.strip()
    
    @log_function_call
    def segment_text(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        try:
            if self.segmenter == 'thulac' and self.thulac_model and self.thulac:
                # THULAC分词
                result = self.thulac_model.cut(text)
                words = [word for word, _ in result]
            else:
                # jieba分词
                self._import_jieba()
                if self.jieba:
                    words = list(self.jieba.cut(text))
                else:
                    # 如果jieba也导入失败，使用简单分词
                    words = list(text)
            
            return words
        except Exception as e:
            log_error(e, "分词失败")
            # 回退到简单分词
            return list(text)
    
    @log_function_call
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        去除停用词
        
        Args:
            words: 词语列表
            
        Returns:
            去除停用词后的词语列表
        """
        return [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
    
    @log_function_call
    def preprocess(self, text: str) -> List[str]:
        """
        完整的文本预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的词语列表
        """
        self.logger.info(f"开始预处理文本: {text[:50]}...")
        
        # 长度检查
        if not text or len(text) < config['default'].PREPROCESSING_CONFIG.get('min_text_length', 5):
            self.logger.warning(f"文本长度不足: {len(text) if text else 0}")
            return []
        
        if len(text) > config['default'].PREPROCESSING_CONFIG.get('max_text_length', 10000):
            text = text[:config['default'].PREPROCESSING_CONFIG.get('max_text_length', 10000)]
        
        # 清洗文本
        cleaned_text = self.clean_text(text)
        self.logger.info(f"清洗后文本: {cleaned_text}")
        if not cleaned_text:
            self.logger.warning("文本清洗后为空")
            return []
        
        # 分词
        words = self.segment_text(cleaned_text)
        self.logger.info(f"分词结果: {words}")
        if not words:
            self.logger.warning("分词结果为空")
            return []
        
        # 去除停用词
        filtered_words = self.remove_stopwords(words)
        self.logger.info(f"去停用词后: {filtered_words}")
        
        return filtered_words
    
    @log_function_call
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            预处理结果列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.preprocess(text)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    log_info(f"已处理 {i + 1}/{len(texts)} 条文本")
                    
            except Exception as e:
                log_error(e, f"处理第{i+1}条文本失败")
                results.append([])
        
        return results
    
    def load_data_from_excel(self, file_path: Union[str, Path], text_column: str = 'content') -> List[str]:
        """
        从Excel文件加载文本数据
        
        Args:
            file_path: Excel文件路径
            text_column: 文本列名
            
        Returns:
            文本列表
        """
        try:
            df = pd.read_excel(file_path)
            if text_column not in df.columns:
                raise ValueError(f"列 '{text_column}' 不存在于Excel文件中")
            
            texts = df[text_column].fillna('').astype(str).tolist()
            log_info(f"从 {file_path} 加载了 {len(texts)} 条文本")
            return texts
            
        except Exception as e:
            log_error(e, f"加载Excel文件失败: {file_path}")
            return []

# 全局预处理器实例
_preprocessor = None

def get_preprocessor(segmenter: str = None) -> TextPreprocessor:
    """获取全局预处理器实例"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor(segmenter)
    return _preprocessor

if __name__ == '__main__':
    # 测试预处理器
    preprocessor = TextPreprocessor()
    
    test_text = "这是一个测试文本，用来检验预处理功能是否正常工作！"
    result = preprocessor.preprocess(test_text)
    print(f"原始文本: {test_text}")
    print(f"预处理结果: {result}")