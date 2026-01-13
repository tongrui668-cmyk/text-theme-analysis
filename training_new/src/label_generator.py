# 标签生成模块

from src.config import config


class LabelGenerator:
    """业务标签生成器"""
    
    def __init__(self):
        """初始化标签生成器"""
        # 主题类型定义，包含更详细的关键词分类
        self.topic_categories = {
            '社交互动': {
                'keywords': ['社交', '线下', '圈子', '社区', '互动', '交流', '交友', '聚会', '活动'],
                'subtypes': ['线下社交', '社区互动', '交友体验']
            },
            '匹配推荐': {
                'keywords': ['匹配', '推荐', '结果', '质量', '精准', '算法', '推荐 算法'],
                'subtypes': ['匹配质量', '推荐效果', '算法精准度']
            },
            '脱单恋爱': {
                'keywords': ['脱单', '恋爱', '男朋友', '女朋友', '结婚', '对象', '谈恋爱'],
                'subtypes': ['脱单效果', '恋爱体验', '婚恋需求']
            },
            '用户质量': {
                'keywords': ['用户', '质量', '正常人', '真实', '靠谱', '真诚', '优质'],
                'subtypes': ['用户质量', '真实性评估', '用户画像']
            },
            '功能体验': {
                'keywords': ['功能', 'app', 'AI', '探探', '抖音', '软件', '界面', '操作'],
                'subtypes': ['功能评价', '界面体验', '技术功能']
            },
            '使用反馈': {
                'keywords': ['体验', '好用', '满意', '卸载', '失望', '推荐', '不推荐'],
                'subtypes': ['使用体验', '用户满意度', '留存分析']
            },
            '平台对比': {
                'keywords': ['美团', 'soul', '二狗', '青藤', 'tinder', '探探', '对比', '竞品'],
                'subtypes': ['平台对比', '竞品分析', '市场定位']
            },
            '会员服务': {
                'keywords': ['会员', '付费', '免费', '价格', '费用', '性价比'],
                'subtypes': ['会员服务', '付费体验', '价格策略']
            }
        }
        
        # 避免标签重复的计数器
        self.label_counter = {}
        
        # 加载停用词列表
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self):
        """从文件加载停用词列表"""
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
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
            '会', '着', '没有', '看', '好', '自己', '这'
        ])
    
    def _filter_stopwords(self, words):
        """
        过滤关键词中的停用词
        
        Args:
            words: 关键词列表
            
        Returns:
            list: 过滤后的关键词列表
        """
        return [word for word in words if word not in self.stopwords]
    
    def get_business_labels(self, n_topics, lda_model=None, vectorizer=None):
        """
        基于关键词语义自动生成业务化标签
        
        Args:
            n_topics: 主题数量
            lda_model: LDA模型（用于提取关键词）
            vectorizer: 向量化器（用于获取特征名称）
            
        Returns:
            dict: 业务标签字典，格式：{主题编号: 业务标签}
        """
        business_labels = {}
        
        # 重置标签计数器
        self.label_counter = {}
        
        # 如果提供了模型和向量化器，基于关键词自动生成标签
        if lda_model and vectorizer:
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx in range(n_topics):
                # 获取主题的前15个关键词
                topic = lda_model.components_[topic_idx]
                top_words = [feature_names[i] for i in topic.argsort()[:-15:-1]]
                
                # 基于关键词语义生成标签
                label = self._generate_label_from_keywords(top_words, topic_idx)
                # 确保标签唯一性
                unique_label = self._ensure_label_uniqueness(label)
                business_labels[topic_idx] = unique_label
        else:
            #  fallback: 使用通用标签
            for i in range(n_topics):
                label = f'主题{i+1}分析'
                business_labels[i] = self._ensure_label_uniqueness(label)
        
        return business_labels
    
    def _generate_label_from_keywords(self, top_words, topic_idx):
        """
        从关键词生成标签

        Args:
            top_words: 主题的关键词列表
            topic_idx: 主题编号

        Returns:
            str: 生成的业务标签
        """
        # 1. 过滤掉停用词，只保留有意义的关键词
        filtered_words = self._filter_stopwords(top_words)
        
        # 2. 识别主题类别
        category_scores = {}
        
        for category, info in self.topic_categories.items():
            score = sum(1 for keyword in info['keywords'] if keyword in filtered_words)
            if score > 0:
                category_scores[category] = score
        
        # 3. 选择得分最高的类别
        if category_scores:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
            primary_info = self.topic_categories[primary_category]
            
            # 4. 基于关键词选择最合适的子类型
            subtype = self._select_most_relevant_subtype(filtered_words, primary_info['subtypes'])
            
            # 5. 分析关键词，添加具体维度
            specific_dimension = self._extract_specific_dimension(filtered_words)
            
            # 6. 提取额外的区分性关键词
            differentiator = self._extract_differentiator(filtered_words, primary_category)
            
            # 7. 组合标签
            if specific_dimension:
                label = f'{subtype}与{specific_dimension}'
            elif differentiator:
                label = f'{subtype}({differentiator})'
            else:
                label = subtype
        else:
            # 如果没有匹配到类别，基于关键词生成描述性标签
            label = self._generate_descriptive_label(filtered_words, topic_idx)
        
        return label
    
    def _select_most_relevant_subtype(self, top_words, subtypes):
        """
        基于关键词选择最合适的子类型

        Args:
            top_words: 过滤后的关键词列表
            subtypes: 子类型列表

        Returns:
            str: 最相关的子类型
        """
        # 子类型关键词映射，添加更多区分性关键词
        subtype_keywords = {
            '线下社交': ['线下', '社交', '约会', '见面', '活动', '聚会', '面基'],
            '社区互动': ['社区', '圈子', '互动', '交流', '论坛', '群组'],
            '交友体验': ['交友', '朋友', '认识', '接触', '了解', '熟悉'],
            '匹配质量': ['匹配', '质量', '精准', '结果', '成功', '有效'],
            '推荐效果': ['推荐', '效果', '算法', 'AI', '智能', '个性化'],
            '算法精准度': ['算法', '精准', 'AI', '技术', '模型', '数据'],
            '脱单效果': ['脱单', '对象', '成功', '恋爱', '找到', '牵手'],
            '恋爱体验': ['恋爱', '体验', '关系', '相处', '感情', '约会'],
            '婚恋需求': ['结婚', '婚恋', '需求', '长期', '稳定', '伴侣'],
            '用户质量': ['用户', '质量', '真实', '靠谱', '真诚', '优质', '正常人'],
            '真实性评估': ['真实', '评估', '验证', '审核', '认证', '核实'],
            '用户画像': ['用户', '画像', '特征', '标签', '属性', '分析'],
            '功能评价': ['功能', '评价', '体验', '使用', '操作', '特性'],
            '界面体验': ['界面', '体验', '设计', '操作', '视觉', '交互'],
            '技术功能': ['技术', '功能', 'AI', '智能', '算法', '创新'],
            '使用体验': ['使用', '体验', '感受', '评价', '反馈', '心得'],
            '用户满意度': ['满意', '满意度', '评价', '推荐', '好评', '认可'],
            '留存分析': ['留存', '分析', '活跃', '长期', '持续', '忠诚'],
            '平台对比': ['平台', '对比', '竞品', '其他', '比较', '不同'],
            '竞品分析': ['竞品', '分析', '对比', '市场', '竞争', '对手'],
            '市场定位': ['市场', '定位', '目标', '用户', '需求', '策略'],
            '会员服务': ['会员', '服务', '付费', '权益', '特权', '等级'],
            '付费体验': ['付费', '体验', '价格', '费用', '成本', '支出'],
            '价格策略': ['价格', '策略', '费用', '性价比', '优惠', '折扣']
        }

        # 计算每个子类型的匹配分数
        best_score = 0
        best_subtype = subtypes[0]  # 默认使用第一个子类型

        for subtype in subtypes:
            if subtype in subtype_keywords:
                score = sum(1 for keyword in subtype_keywords[subtype] if keyword in top_words)
                if score > best_score:
                    best_score = score
                    best_subtype = subtype

        return best_subtype

    def _extract_differentiator(self, top_words, primary_category):
        """
        提取区分性关键词，用于区分同类别的不同主题

        Args:
            top_words: 过滤后的关键词列表
            primary_category: 主要类别

        Returns:
            str: 区分性描述
        """
        # 类别特定的区分关键词，添加更多有意义的区分词
        differentiators = {
            '社交互动': ['圈子', '社区', '约会', '交友', '活动', '聚会', '面基', '论坛'],
            '匹配推荐': ['AI', '算法', '推荐', '精准', '智能', '个性化', '技术', '数据'],
            '脱单恋爱': ['恋爱', '结婚', '对象', '相处', '感情', '牵手', '成功', '找到'],
            '用户质量': ['真实', '靠谱', '优质', '真诚', '正常人', '验证', '审核', '认证'],
            '功能体验': ['界面', '操作', '技术', '智能', '特性', '设计', '交互', '视觉'],
            '使用反馈': ['满意', '失望', '推荐', '卸载', '体验', '感受', '心得', '评价'],
            '平台对比': ['美团', 'soul', '二狗', '探探', 'tinder', '青藤', '竞品', '对手'],
            '会员服务': ['付费', '免费', '价格', '性价比', '权益', '特权', '等级', '优惠']
        }

        if primary_category in differentiators:
            for keyword in differentiators[primary_category]:
                if keyword in top_words:
                    return keyword
        
        # 如果没有找到类别特定的区分词，尝试从过滤后的关键词中提取
        meaningful_words = [word for word in top_words if word not in self.stopwords]
        if meaningful_words:
            return meaningful_words[0]

        return ''

    def _extract_specific_dimension(self, top_words):
        """
        从关键词中提取具体维度

        Args:
            top_words: 过滤后的关键词列表

        Returns:
            str: 具体维度描述
        """
        # 具体维度关键词，添加更多维度和关键词
        dimensions = {
            '用户留存': ['留存', '活跃', '持续', '长期', '忠诚', '活跃用户'],
            '满意度': ['满意', '喜欢', '推荐', '好评', '认可', '赞赏'],
            '服务质量': ['服务', '客服', '支持', '帮助', '售后', '响应'],
            '价格策略': ['价格', '费用', '性价比', '优惠', '折扣', '成本'],
            '技术体验': ['技术', '算法', '智能', 'AI', '功能', '创新'],
            '内容质量': ['内容', '质量', '丰富', '优质', '有价值', '信息'],
            '使用便捷': ['便捷', '简单', '易用', '方便', '操作', '界面'],
            '安全隐私': ['安全', '隐私', '保护', '保密', '安全问题', '隐私保护'],
            '社交互动': ['社交', '互动', '交流', '圈子', '社区', '交友'],
            '匹配效果': ['匹配', '效果', '精准', '推荐', '成功', '找到'],
            '用户质量': ['用户', '质量', '真实', '靠谱', '真诚', '正常人']
        }

        for dimension, keywords in dimensions.items():
            if any(keyword in top_words for keyword in keywords):
                return dimension

        return ''
    
    def _generate_descriptive_label(self, top_words, topic_idx):
        """
        生成描述性标签

        Args:
            top_words: 过滤后的关键词列表
            topic_idx: 主题编号

        Returns:
            str: 描述性标签
        """
        # 从过滤后的关键词中提取最具代表性的词
        representative_words = [word for word in top_words if word not in self.stopwords][:3]
        
        # 组合成标签
        if representative_words:
            label = f'{"-".join(representative_words)}相关分析'
        else:
            label = f'主题{topic_idx+1}分析'
        
        return label
    
    def _ensure_label_uniqueness(self, label):
        """
        确保标签唯一性

        Args:
            label: 原始标签

        Returns:
            str: 唯一标签
        """
        if label in self.label_counter:
            self.label_counter[label] += 1
            return f'{label}({self.label_counter[label]})'
        else:
            self.label_counter[label] = 0
            return label

