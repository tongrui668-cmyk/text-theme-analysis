import re
try:
    import thulac
    THULAC_AVAILABLE = True
except ImportError:
    THULAC_AVAILABLE = False
import jieba
import os

class DataPreprocessor:
    def __init__(self, stop_words_path=None):
        """
        初始化数据预处理器
        :param stop_words_path: 停用词文件路径
        """
        self.stop_words = set()
        if stop_words_path and os.path.exists(stop_words_path):
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip() for line in f if line.strip())
        
        # 初始化分词器，优先使用thulac，如果失败则使用jieba
        if THULAC_AVAILABLE:
            try:
                self.thu = thulac.thulac(seg_only=True)
                self.use_thulac = True
            except:
                self.use_thulac = False
                jieba.initialize()
        else:
            self.use_thulac = False
            jieba.initialize()
    
    def clean_text(self, text):
        """
        清理文本，去除特殊字符和标点
        :param text: 输入文本
        :return: 清理后的文本
        """
        if not isinstance(text, str):
            return ''
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 去除数字（保留中文数字）
        text = re.sub(r'\d+', '', text)
        
        # 去除英文字母
        text = re.sub(r'[a-zA-Z]', '', text)
        
        # 去除特殊标点符号，保留中文标点
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]', '', text)
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text):
        """
        对文本进行分词
        :param text: 输入文本
        :return: 分词后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        try:
            if self.use_thulac:
                # 使用thulac分词
                words = self.thu.cut(text, text=True)
                return ' '.join(words.split())
            else:
                # 使用jieba分词
                words = jieba.cut(text)
                return ' '.join(words)
        except:
            # 如果分词失败，返回按空格分割的结果
            return ' '.join(text.split())
    
    def remove_stop_words(self, text):
        """
        去除停用词
        :param text: 输入文本（已分词）
        :return: 去除停用词后的文本
        """
        if not isinstance(text, str):
            return ''
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """
        完整的文本预处理流程
        :param text: 输入文本
        :return: 预处理后的文本
        """
        # 清理文本
        cleaned_text = self.clean_text(text)
        
        # 分词
        segmented_text = self.segment_text(cleaned_text)
        
        # 去除停用词
        processed_text = self.remove_stop_words(segmented_text)
        
        return processed_text