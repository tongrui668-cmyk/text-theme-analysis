#!/usr/bin/env python3
"""
增强的文本预处理模块
包含更全面的预处理功能
"""

import re
import jieba
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTextPreprocessor:
    """增强的文本预处理类"""
    
    def __init__(self):
        """初始化文本预处理器"""
        # 扩展停用词列表
        self.stopwords = self._load_enhanced_stopwords()
        # 加载自定义词典
        self._load_custom_dict()
        # 初始化语义增强词典
        self.semantic_enhancements = self._load_semantic_enhancements()
        # 初始化上下文规则
        self.context_rules = self._load_context_rules()
    
    def _load_enhanced_stopwords(self):
        """从文件加载增强的停用词列表"""
        stopwords = set()
        stopwords_file = '../data/processed/stopwords.txt'
        
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 跳过注释行
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 按空格分割词
                        words = line.split()
                        stopwords.update(words)
        except Exception as e:
            print(f"加载停用词文件失败: {e}")
            # 如果加载失败，使用默认停用词
            stopwords = self._get_default_stopwords()
        
        return stopwords
    
    def _get_default_stopwords(self):
        """获取默认停用词列表（当文件加载失败时使用）"""
        return set([
            # 基础停用词
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
            '会', '着', '没有', '看', '好', '自己', '这', '只不过', '只是', '就是', '还是', '但是', '可是', '不过',
            
            # 社交媒体停用词
            '哦', '嗯', '啊', '呀', '呢', '吧', '啦', '哟', '嗬', '嘿', '哈', '哎', '唉', '嗯哼',
            
            # 语气词
            '哈哈', '哈哈哈哈', '呵呵', '嘿嘿', '呜呜', '啊啊', '嘻嘻', '咯咯', '呵呵呵', '哈哈哈', '呜呜呜', '啊啊啊',
            
            # 社交软件特有词汇
            'app', 'APP', '软件', '平台', '应用', '程序', '工具', '系统', '官网', '官方', '版本', '更新', '下载', '安装',
            
            # 无意义词汇
            '什么', '怎么', '这样', '那样', '这个', '那个', '这些', '那些', '这里', '那里',
            '可以', '不能', '不会', '不是', '没有', '不要', '不用', '不能不', '不得不',
            '觉得', '感觉', '认为', '希望', '想', '要', '应该', '可能', '也许', '大概',
            
            # 高频无效词（根据实际输出添加）
            '很多', '知道', '真的', '好像', '几乎', '每一', '还有', '已经', '上面', '时候',
            '一下', '大家', '然后', '之前', '开始', '所以', '出来', '这种', '干嘛',
            '不到', '厉害', '看看', '不了', '别人', '有人', '里面', '不如', 
             'ai', '这么',
            '好多',  '是不是', '实战', '剩下',
             '这种','确实', 
            
            # 数字和英文
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            
            # 网络用语
            '666', '233', 'hhh', 'qwq', 'qaq', 'awsl', 'yyds', 'nb', '牛批', '牛逼',
            '佛系', '躺平', '内卷', 'emo', '破防', '蚌埠住了', '绝绝子', 'YYDS',
            
            # 时间相关
            '今天', '明天', '昨天', '现在', '过去', '将来', '最近', '以前', '以后',
            '早上', '中午', '晚上', '白天', '黑夜', '周末', '假期', '工作日'
        ])
    
    def _load_custom_dict(self):
        """加载自定义词典"""
        # 尝试加载自定义词典
        custom_dict_path = '../training/custom_dict.txt'
        try:
            jieba.load_userdict(custom_dict_path)
            logger.info(f"加载自定义词典: {custom_dict_path}")
        except Exception as e:
            logger.warning(f"加载自定义词典失败: {e}")
    
    def clean_text(self, text):
        """
        增强的文本清洗
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 全角转半角
        text = self._full_to_half(text)
        
        # 2. 统一大小写（转为小写）
        text = text.lower()
        
        # 3. 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 4. 移除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 5. 移除电话号码
        text = re.sub(r'1[3-9]\d{9}', '', text)
        
        # 6. 移除社交媒体标签
        text = re.sub(r'@[^\s]+', '', text)
        
        # 7. 移除话题标签
        text = re.sub(r'#([^#\s]+)#', '', text)
        text = re.sub(r'#[^#\s]+', '', text)
        
        # 8. 移除表情符号（Unicode范围）
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text, flags=re.UNICODE)
        
        # 9. 移除特殊字符和多余空格
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 10. 移除连续重复字符（如"哈哈哈哈"）
        text = re.sub(r'(.)\1{3,}', r'\1\1', text)
        
        # 11. 移除连续重复短语
        text = re.sub(r'(\b[\u4e00-\u9fa5]+\b)\s+\1', r'\1', text)
        
        return text.strip()
    
    def _full_to_half(self, text):
        """
        全角转半角
        
        Args:
            text: 全角文本
            
        Returns:
            str: 半角文本
        """
        result = []
        for char in text:
            code = ord(char)
            if code == 0x3000:
                code = 0x0020
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result.append(chr(code))
        return ''.join(result)
    
    def tokenize(self, text):
        """
        增强的分词（支持中英文混合）
        
        Args:
            text: 清洗后的文本
            
        Returns:
            list: 分词结果
        """
        if not text:
            return []
        
        # 使用jieba分词，添加一些分词参数
        words = list(jieba.cut(text, cut_all=False, HMM=True))
        
        # 过滤空词
        words = [word.strip() for word in words if word.strip()]
        
        # 处理英文单词和数字
        processed_words = []
        for word in words:
            if word.isalpha() and all(ord(c) < 128 for c in word):
                # 保留英文单词
                processed_words.append(word)
            elif word.isdigit():
                # 数字作为单独的词
                processed_words.append(word)
            elif any(ord(c) >= 128 for c in word):
                # 中文词汇
                processed_words.append(word)
            else:
                # 其他字符
                processed_words.append(word)
        
        return processed_words
    
    def remove_duplicate_phrases(self, words):
        """
        增强的重复短语移除
        
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
        增强的停用词移除（支持中英文混合）
        
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
            
            # 跳过长度为1的无意义词（但保留一些有意义的单字词）
            meaningful_single_chars = {'好', '坏', '大', '小', '多', '少', '高', '低', '强', '弱', '真', '假'}
            if len(word) <= 1 and word not in meaningful_single_chars and not word.isdigit():
                continue
            
            # 跳过过长的词（可能是噪声）
            if len(word) > 10:
                continue
            
            # 保留英文单词（长度大于1）
            if word.isalpha() and all(ord(c) < 128 for c in word) and len(word) > 1:
                filtered_words.append(word)
            # 保留中文词汇
            elif any(ord(c) >= 128 for c in word):
                filtered_words.append(word)
            # 保留数字
            elif word.isdigit():
                filtered_words.append(word)
        
        return filtered_words
    
    def _load_context_rules(self):
        """加载上下文规则"""
        return {
            # 多义词上下文规则
            '安全': {
                '约会': '线下安全',
                '见面': '线下安全',
                '线下': '线下安全',
                '隐私': '隐私安全',
                '信息': '信息安全',
                '数据': '数据安全'
            },
            '匹配': {
                '算法': '推荐匹配',
                '智能': '推荐匹配',
                'AI': '推荐匹配',
                '对象': '伴侣匹配',
                '伴侣': '伴侣匹配',
                '脱单': '伴侣匹配'
            },
            '体验': {
                '用户': '用户体验',
                '使用': '使用体验',
                '产品': '产品体验',
                '服务': '服务体验'
            },
            '质量': {
                '用户': '用户质量',
                '内容': '内容质量',
                '服务': '服务质量',
                '产品': '产品质量'
            },
            '功能': {
                '产品': '产品功能',
                '社交': '社交功能',
                '推荐': '推荐功能',
                '安全': '安全功能'
            },
            '社交': {
                '软件': '社交软件',
                '平台': '社交平台',
                '互动': '社交互动',
                '活动': '社交活动'
            }
        }
    
    def _load_semantic_enhancements(self):
        """加载语义增强词典"""
        return {
            # 核心社交目标
            '脱单': '寻找伴侣',
            '恋爱': '寻找伴侣',
            '相亲': '寻找伴侣',
            '结婚': '寻找伴侣',
            '交友': '寻找伴侣',
            '搭子': '寻找伴侣',
            '伴侣': '寻找伴侣',
            '对象': '寻找伴侣',
            
            # 推荐系统相关
            '匹配': '推荐算法',
            '推荐': '推荐算法',
            '算法': '推荐算法',
            '智能': '推荐算法',
            'AI': '推荐算法',
            '个性化': '推荐算法',
            '精准': '推荐算法',
            '智能匹配': '推荐算法',
            
            # 社交互动
            '聊天': '社交互动',
            '社交': '社交互动',
            '交流': '社交互动',
            '沟通': '社交互动',
            '互动': '社交互动',
            '对话': '社交互动',
            '连麦': '社交互动',
            '视频聊天': '社交互动',
            
            # 线下活动
            '线下': '线下约会',
            '见面': '线下约会',
            '约会': '线下约会',
            '面基': '线下约会',
            '线下活动': '线下约会',
            '线下见面': '线下约会',
            
            # 用户相关
            '质量': '用户质量',
            '用户': '用户质量',
            '人群': '用户质量',
            '人': '用户质量',
            '用户群体': '用户质量',
            '用户画像': '用户质量',
            '用户分层': '用户质量',
            
            # 安全问题
            '骗子': '安全问题',
            '骗': '安全问题',
            '安全': '安全问题',
            '隐私': '安全问题',
            '隐私保护': '安全问题',
            '诈骗': '安全问题',
            '风险': '安全问题',
            '举报': '安全问题',
            
            # 产品功能
            '功能': '产品功能',
            '特色': '产品功能',
            '设计': '产品功能',
            '界面': '产品功能',
            'UI': '产品功能',
            '交互': '产品功能',
            '操作': '产品功能',
            '流程': '产品功能',
            
            # 用户体验
            '体验': '用户体验',
            '感受': '用户体验',
            '使用': '用户体验',
            '效果': '用户体验',
            '效率': '用户体验',
            '速度': '用户体验',
            '卡顿': '用户体验',
            '流畅': '用户体验',
            '便捷': '用户体验',
            '方便': '用户体验',
            
            # 商业模式
            '会员': '商业模式',
            '付费': '商业模式',
            '免费': '商业模式',
            '收费': '商业模式',
            '订阅': '商业模式',
            '增值服务': '商业模式',
            '广告': '商业模式',
            
            # 内容相关
            '动态': '内容生态',
            '发布': '内容生态',
            '分享': '内容生态',
            '图片': '内容生态',
            '视频': '内容生态',
            '文字': '内容生态',
            '内容': '内容生态',
            
            # 社区相关
            '社区': '社区氛围',
            '氛围': '社区氛围',
            '文化': '社区氛围',
            '规则': '社区氛围',
            '管理': '社区氛围',
            '活跃': '社区氛围',
            '互动': '社区氛围'
        }
    
    def normalize_social_terms(self, words):
        """
        社交媒体用语标准化
        
        Args:
            words: 分词结果
            
        Returns:
            list: 标准化后的词列表
        """
        normalized_words = []
        term_mappings = {
            # 社交软件品牌
            '探探': '社交软件',
            'soul': '社交软件',
            'tinder': '社交软件',
            '牵手': '社交软件',
            '青藤': '社交软件',
            '二狗': '社交软件',
            'boss': '社交软件',
            '小红书': '社交软件',
            '陌陌': '社交软件',
            '微信': '社交软件',
            'qq': '社交软件',
            '抖音': '社交软件',
            '快手': '社交软件',
            '知乎': '社交软件',
            '微博': '社交软件',
            '豆瓣': '社交软件',
            '贴吧': '社交软件',
            'ins': '社交软件',
            'instagram': '社交软件',
            'facebook': '社交软件',
            'twitter': '社交软件',
            'b站': '社交软件',
            '哔哩哔哩': '社交软件',
            'linkedin': '社交软件',
            '领英': '社交软件',
            'whatsapp': '社交软件',
            'telegram': '社交软件',
            'line': '社交软件',
            'kakaotalk': '社交软件',
            'skype': '社交软件',
            'discord': '社交软件',
            
            # 社交功能术语
            'swipe': '滑动匹配',
            '滑动': '滑动匹配',
            '左滑': '滑动匹配',
            '右滑': '滑动匹配',
            '匹配成功': '滑动匹配',
            'match': '滑动匹配',
            
            # 社交行为术语
            'ghosting': '社交行为',
            ' ghost': '社交行为',
            '撩': '社交行为',
            '搭讪': '社交行为',
            '约会软件': '社交软件',
            '交友软件': '社交软件',
            '社交平台': '社交软件',
            
            # 技术术语
            'algorithm': '推荐算法',
            '算法推荐': '推荐算法',
            '机器学习': '推荐算法',
            '深度学习': '推荐算法',
            
            # 商业模式术语
            'freemium': '商业模式',
            '免费增值': '商业模式',
            '订阅制': '商业模式',
            '会员制': '商业模式',
            
            # 内容术语
            'feed': '内容流',
            '信息流': '内容流',
            '动态流': '内容流',
            '推荐流': '内容流'
        }
        
        for word in words:
            normalized_word = term_mappings.get(word, word)
            normalized_words.append(normalized_word)
        
        return normalized_words
    
    def get_contextual_meaning(self, word, context_words):
        """
        根据上下文获取词汇的具体含义
        
        Args:
            word: 目标词汇
            context_words: 上下文词汇列表
            
        Returns:
            str: 词汇在当前上下文中的具体含义
        """
        # 检查是否有多义词上下文规则
        if word in self.context_rules:
            rules = self.context_rules[word]
            
            # 检查上下文是否包含触发词
            for context_word, meaning in rules.items():
                if context_word in context_words:
                    return meaning
        
        # 如果没有上下文规则或未匹配到，返回默认语义增强
        return self.semantic_enhancements.get(word, word)
    
    def enhance_semantics(self, words):
        """
        语义增强处理（上下文感知）
        
        Args:
            words: 分词结果
            
        Returns:
            list: 语义增强后的词列表
        """
        enhanced_words = []
        
        for word in words:
            # 添加原始词
            enhanced_words.append(word)
            
            # 根据上下文获取具体含义
            contextual_meaning = self.get_contextual_meaning(word, words)
            
            # 添加上下文感知的语义增强词
            if contextual_meaning != word and contextual_meaning not in enhanced_words:
                enhanced_words.append(contextual_meaning)
        
        return enhanced_words
    
    def preprocess(self, text):
        """
        增强的文本预处理流程
        
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
        
        # 4. 社交媒体用语标准化
        normalized_words = self.normalize_social_terms(unique_words)
        logger.info(f"标准化后: {normalized_words}")
        
        # 5. 语义增强
        enhanced_words = self.enhance_semantics(normalized_words)
        logger.info(f"语义增强后: {enhanced_words}")
        
        # 6. 移除停用词和无意义语气词
        filtered_words = self.remove_stopwords(enhanced_words)
        logger.info(f"去停用词后: {filtered_words}")
        
        # 7. 再次去重（语义增强可能引入重复）
        final_words = self.remove_duplicate_phrases(filtered_words)
        logger.info(f"最终结果: {final_words}")
        
        return final_words