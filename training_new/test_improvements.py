#!/usr/bin/env python3
"""
测试脚本，验证文本预处理和标签生成的改进效果
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_preprocessor_enhanced import EnhancedTextPreprocessor
from src.label_generator import LabelGenerator

def test_text_preprocessing():
    """测试文本预处理效果"""
    preprocessor = EnhancedTextPreprocessor()
    
    # 测试用例：包含商业推广内容的文本
    test_cases = [
        "这个APP下载讨论区有很多微商客源，还有工作机会，大家可以看看",
        "婚恋交友服务，代理合作，引流推广，赚钱变现，欢迎咨询",
        "很多人都知道，这个平台真的好像几乎每一还有已经上面时候一下大家遇到然后之前开始所以出来这种干嘛不到朋友厉害看看不了别人有人里面不如好玩内容分享注册男生学习恋爱不错ai美团这么好多真人通讯录花钱是不是实战剩下搭子纯圈安全问题男生这种没人男人确实认识",
        "探探APP下载，推荐算法真的很精准，匹配质量高，脱单效果好",
        "线下社交活动，社区互动，交友体验很棒，推荐给大家"
    ]
    
    print("=" * 80)
    print("测试文本预处理效果")
    print("=" * 80)
    
    for i, test_text in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"原始文本: {test_text}")
        
        # 预处理文本
        processed_words = preprocessor.preprocess(test_text)
        print(f"预处理后: {processed_words}")
        
        # 计算过滤掉的词数
        original_words = len(test_text.split())
        processed_count = len(processed_words)
        filtered_count = original_words - processed_count
        print(f"过滤掉的词数: {filtered_count}")

def test_label_generation():
    """测试标签生成效果"""
    label_generator = LabelGenerator()
    
    # 测试用例：不同主题的关键词列表
    test_cases = [
        ["社交", "线下", "圈子", "社区", "互动", "交流", "交友", "聚会", "活动"],
        ["匹配", "推荐", "结果", "质量", "精准", "算法", "推荐算法"],
        ["脱单", "恋爱", "男朋友", "女朋友", "结婚", "对象", "谈恋爱"],
        ["用户", "质量", "正常人", "真实", "靠谱", "真诚", "优质"],
        ["功能", "app", "AI", "探探", "抖音", "软件", "界面", "操作"],
        ["体验", "好用", "满意", "卸载", "失望", "推荐", "不推荐"],
        ["美团", "soul", "二狗", "青藤", "tinder", "探探", "对比", "竞品"],
        ["会员", "付费", "免费", "价格", "费用", "性价比"]
    ]
    
    print("\n" + "=" * 80)
    print("测试标签生成效果")
    print("=" * 80)
    
    for i, keywords in enumerate(test_cases):
        # 生成标签
        label = label_generator._generate_label_from_keywords(keywords, i)
        unique_label = label_generator._ensure_label_uniqueness(label)
        
        print(f"\n测试用例 {i+1}:")
        print(f"关键词: {keywords}")
        print(f"生成标签: {label}")
        print(f"唯一标签: {unique_label}")

if __name__ == "__main__":
    test_text_preprocessing()
    test_label_generation()
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)