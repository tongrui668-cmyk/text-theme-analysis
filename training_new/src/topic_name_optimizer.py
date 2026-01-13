# 主题名称自动优化模块

import re
from collections import Counter

class TopicNameOptimizer:
    """主题名称自动优化器"""
    
    def __init__(self):
        """初始化主题名称优化器"""
        # 主题类别关键词映射
        self.topic_categories = {
            '社交互动': ['社交', '互动', '交流', '聊天', '认识', '朋友', '搭子'],
            '线下社交': ['线下', '见面', '约会', '交友', '聚会', '活动'],
            '脱单恋爱': ['脱单', '恋爱', '对象', '找对象', '伴侣', '寻找伴侣'],
            '用户质量': ['用户', '质量', '真实', '靠谱', '真诚', '优质'],
            '会员服务': ['会员', '服务', '付费', '免费', '商业模式'],
            '平台对比': ['平台', '对比', '其他', '竞品', '软件', 'app'],
            '使用体验': ['体验', '好用', '满意', '喜欢', '推荐', '卸载'],
            '匹配推荐': ['匹配', '推荐', '算法', '结果', '精准']
        }
        
        # 质量评估相关词汇
        self.quality_terms = ['满意度', '质量', '体验', '效果']
        
        # 模糊词汇（需要避免的）
        self.vague_terms = ['好像', '相关', '分析', '这种', '然后']
    
    def optimize_topic_name(self, topic_name, keywords):
        """
        优化单个主题名称
        
        Args:
            topic_name: 当前主题名称
            keywords: 主题的关键词列表
            
        Returns:
            str: 优化后的主题名称
        """
        # 1. 清理当前主题名称
        cleaned_name = self._clean_topic_name(topic_name)
        
        # 2. 分析关键词，提取核心概念
        core_concepts = self._extract_core_concepts(keywords)
        
        # 3. 生成候选主题名称
        candidate_names = self._generate_candidate_names(cleaned_name, core_concepts, keywords)
        
        # 4. 评估候选名称并选择最佳
        best_name = self._select_best_name(candidate_names, keywords)
        
        return best_name
    
    def _clean_topic_name(self, topic_name):
        """
        清理主题名称，去除重复内容和模糊词汇
        
        Args:
            topic_name: 当前主题名称
            
        Returns:
            str: 清理后的主题名称
        """
        # 去除模糊词汇
        cleaned = topic_name
        for term in self.vague_terms:
            cleaned = cleaned.replace(term, '')
        
        # 去除重复内容（如"社交互动-聊天-聊天 社交互动分析"中的重复部分）
        parts = re.split(r'[-_\s]+', cleaned)
        unique_parts = []
        seen = set()
        for part in parts:
            if part and part not in seen:
                unique_parts.append(part)
                seen.add(part)
        
        # 重新组合
        cleaned = '-'.join(unique_parts)
        
        # 去除多余的标点符号
        cleaned = re.sub(r'[-_\s]+', '-', cleaned)
        cleaned = cleaned.strip('-')
        
        return cleaned
    
    def _extract_core_concepts(self, keywords):
        """
        从关键词中提取核心概念
        
        Args:
            keywords: 主题的关键词列表
            
        Returns:
            list: 核心概念列表
        """
        concept_scores = {}
        
        # 统计每个类别的关键词出现次数
        for category, category_keywords in self.topic_categories.items():
            score = sum(1 for kw in keywords if kw in category_keywords)
            if score > 0:
                concept_scores[category] = score
        
        # 按得分排序，取前2个核心概念
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        core_concepts = [concept for concept, _ in sorted_concepts[:2]]
        
        return core_concepts
    
    def _generate_candidate_names(self, current_name, core_concepts, keywords):
        """
        生成候选主题名称
        
        Args:
            current_name: 当前主题名称
            core_concepts: 核心概念列表
            keywords: 主题的关键词列表
            
        Returns:
            list: 候选主题名称列表
        """
        candidates = []
        
        # 1. 基于核心概念生成名称
        if core_concepts:
            # 单个核心概念
            for concept in core_concepts:
                candidates.append(concept)
            
            # 组合核心概念
            if len(core_concepts) > 1:
                combined = '-'.join(core_concepts)
                candidates.append(combined)
        
        # 2. 基于核心概念和质量评估词汇生成名称
        for concept in core_concepts:
            for quality_term in self.quality_terms:
                candidates.append(f"{concept}与{quality_term}")
        
        # 3. 保留当前名称作为候选
        if current_name:
            candidates.append(current_name)
        
        # 4. 基于前几个关键词生成名称
        if len(keywords) >= 2:
            keyword_based = '-'.join(keywords[:2])
            candidates.append(keyword_based)
        
        # 去重并过滤空值
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        
        return unique_candidates
    
    def _select_best_name(self, candidate_names, keywords):
        """
        评估候选名称并选择最佳
        
        Args:
            candidate_names: 候选主题名称列表
            keywords: 主题的关键词列表
            
        Returns:
            str: 最佳主题名称
        """
        if not candidate_names:
            return "未命名主题"
        
        # 评分标准：
        # 1. 长度适中（3-8个字符）
        # 2. 包含核心概念
        # 3. 与关键词相关性高
        # 4. 简洁明了
        
        scores = []
        for name in candidate_names:
            score = 0
            
            # 长度评分
            length = len(name)
            if 3 <= length <= 8:
                score += 3
            elif 2 <= length <= 10:
                score += 2
            elif length >= 1:
                score += 1
            
            # 核心概念评分
            name_lower = name.lower()
            for category, category_keywords in self.topic_categories.items():
                if category in name:
                    score += 2
                for kw in category_keywords:
                    if kw in name_lower:
                        score += 1
            
            # 关键词相关性评分
            keyword_matches = sum(1 for kw in keywords if kw in name)
            score += keyword_matches * 2
            
            # 简洁性评分（避免过长的名称）
            if length <= 6:
                score += 2
            elif length <= 8:
                score += 1
            
            scores.append((score, name))
        
        # 选择得分最高的
        scores.sort(reverse=True, key=lambda x: x[0])
        best_name = scores[0][1]
        
        return best_name
    
    def optimize_all_topics(self, topic_keywords):
        """
        优化所有主题的名称
        
        Args:
            topic_keywords: 字典，键为当前主题名称，值为关键词列表
            
        Returns:
            dict: 字典，键为优化后的主题名称，值为关键词列表
        """
        optimized_topics = {}
        
        for old_name, keywords in topic_keywords.items():
            new_name = self.optimize_topic_name(old_name, keywords)
            # 确保名称唯一性
            unique_name = self._ensure_unique_name(new_name, list(optimized_topics.keys()))
            optimized_topics[unique_name] = keywords
        
        return optimized_topics
    
    def _ensure_unique_name(self, name, existing_names):
        """
        确保主题名称的唯一性
        
        Args:
            name: 主题名称
            existing_names: 已存在的主题名称列表
            
        Returns:
            str: 唯一的主题名称
        """
        if name not in existing_names:
            return name
        
        # 添加数字后缀以确保唯一性
        counter = 1
        while True:
            candidate = f"{name}({counter})"
            if candidate not in existing_names:
                return candidate
            counter += 1
