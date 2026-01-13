# 文本预处理模块

import re
import jieba
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self):
        """初始化文本预处理器"""
        # 扩展停用词列表
        self.stopwords = set([
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
            '会', '着', '没有', '看', '好', '自己', '这', '只不过', '哈哈', '哈哈哈哈', '呵呵', '嘿嘿', '呜呜', '啊啊', '哦', '嗯', '哎',
            '啊', '吧', '呢', '吗', '呀', '哦', '嗯', '唉', '哎呀', '哎哟', '哎', '嗨', '喂', '嗯哼', '哈哈', '呵呵', '嘻嘻', '嘿嘿', '呜呜',
            '哭哭', '笑笑', '哈哈笑', '呵呵笑', '嘻嘻笑', '嘿嘿笑', '呜呜哭', '哭哭啼啼', '笑笑嘻嘻', '哈哈哈哈哈', '呵呵呵呵呵',
            '这个', '那个', '这些', '那些', '这样', '那样', '这么', '那么', '这么样', '那么样', '这么着', '那么着',
            '可以', '可能', '应该', '必须', '需要', '想要', '希望', '觉得', '认为', '感觉', '感受', '觉得', '认为',
            '时候', '时间', '地方', '位置', '东西', '事情', '事物', '问题', '情况', '状况', '状态', '样子', '模样',
            '但是', '可是', '不过', '然而', '所以', '因此', '因为', '由于', '虽然', '尽管', '即使', '假如', '如果',
            '已经', '曾经', '刚刚', '刚才', '现在', '目前', '将来', '未来', '过去', '以前', '之后', '之后',
            '这里', '那里', '这里', '那里', '这儿', '那儿', '这边', '那边', '这里', '那里',
            '什么', '怎么', '为什么', '何时', '何地', '何人', '何物', '如何', '怎样', '多少', '多久', '多远',
            '非常', '特别', '十分', '很', '相当', '比较', '太', '最', '更', '更加', '越来越', '越', '愈发',
            '只', '就', '才', '都', '也', '还', '又', '再', '就', '才', '都', '也', '还', '又', '再',
            '能', '能够', '可以', '会', '可能', '也许', '或许', '大概', '大约', '左右', '上下', '前后',
            '从', '自', '由', '由', '从', '自从', '自从', '打', '在', '当', '对', '对于', '关于', '至于',
            '把', '将', '使', '让', '令', '叫', '被', '给', '为', '替', '帮', '帮', '给',
            '而', '且', '与', '和', '及', '以及', '并', '并且', '不但', '不仅', '不止', '不只', '不但', '不仅',
            '了', '着', '过', '的', '地', '得', '所', '的', '地', '得', '所', '的', '地', '得',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', '零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖', '拾'
        ])
    
    def clean_text(self, text):
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        # 移除特殊字符和多余空格
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        return text.strip()
    
    def tokenize(self, text):
        """
        分词
        
        Args:
            text: 清洗后的文本
            
        Returns:
            list: 分词结果
        """
        return list(jieba.cut(text))
    
    def remove_duplicate_phrases(self, words):
        """
        移除重复短语
        
        Args:
            words: 分词结果
            
        Returns:
            list: 去重后的词列表
        """
        if not words:
            return []
        
        unique_words = []
        seen = set()
        
        for word in words:
            # 跳过空词和已见过的词
            if word and word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        return unique_words
    
    def _is_repetitive_interjection(self, word):
        """
        判断是否为重复的语气词
        
        Args:
            word: 词语
            
        Returns:
            bool: 是否为重复语气词
        """
        # 检查是否由相同字符组成且长度大于2
        if len(word) > 2:
            if all(char == word[0] for char in word):
                return True
        return False
    
    def remove_stopwords(self, words):
        """
        移除停用词和无意义的语气词
        
        Args:
            words: 分词结果
            
        Returns:
            list: 去停用词后的词列表
        """
        filtered_words = []
        
        for word in words:
            # 跳过停用词
            if word in self.stopwords:
                continue
            # 跳过重复的语气词
            if self._is_repetitive_interjection(word):
                continue
            # 跳过长度为1的无意义词
            if len(word) <= 1 and not word.isdigit():
                continue
            
            filtered_words.append(word)
        
        return filtered_words
    
    def preprocess(self, text):
        """
        完整的文本预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            list: 预处理后的词列表
        """
        logger.info(f"开始预处理文本: {text[:50]}...")
        
        # 1. 清洗文本
        cleaned_text = self.clean_text(text)
        logger.info(f"清洗后文本: {cleaned_text}")
        
        # 2. 分词
        words = self.tokenize(cleaned_text)
        logger.info(f"分词结果: {words}")
        
        # 3. 移除重复短语
        unique_words = self.remove_duplicate_phrases(words)
        logger.info(f"去重复短语后: {unique_words}")
        
        # 4. 移除停用词和无意义语气词
        filtered_words = self.remove_stopwords(unique_words)
        logger.info(f"去停用词后: {filtered_words}")
        
        return filtered_words
