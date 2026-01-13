#!/usr/bin/env python3
"""
文本预处理效果比较测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.text_preprocessor import TextPreprocessor
from src.text_preprocessor_enhanced import EnhancedTextPreprocessor

def test_preprocessors():
    """测试并比较两个预处理器的效果"""
    
    # 初始化预处理器
    original_preprocessor = TextPreprocessor()
    enhanced_preprocessor = EnhancedTextPreprocessor()
    
    # 测试文本
    test_texts = [
        "这个APP真的很好用，推荐给大家！哈哈哈哈",
        "探探、soul这些社交软件，哪个更好用？",
        "为什么我在上面找不到合适的对象呢？",
        "会员功能太贵了，不如用免费的功能",
        "线下见面感觉不错，但是软件上的照片和本人差别太大了",
        "小红书上推荐的这个软件，真的有那么好吗？",
        "1234567890abcdefghijklmnopqrstuvwxyz",
        "http://example.com 这是一个测试链接",
        "用户体验很差，界面设计不好看，功能也不实用",
        "脱单效果不错，已经找到了合适的对象"
    ]
    
    print("=== 文本预处理效果比较 ===\n")
    
    for i, text in enumerate(test_texts):
        print(f"测试文本 {i+1}: {text}")
        
        # 原始预处理器
        original_result = original_preprocessor.preprocess(text)
        print(f"原始预处理器: {original_result}")
        
        # 增强预处理器
        enhanced_result = enhanced_preprocessor.preprocess(text)
        print(f"增强预处理器: {enhanced_result}")
        
        print(f"词数变化: {len(original_result)} → {len(enhanced_result)}")
        print()
    
    # 性能测试
    import time
    
    print("=== 性能测试 ===")
    start_time = time.time()
    for text in test_texts * 100:
        original_preprocessor.preprocess(text)
    original_time = time.time() - start_time
    print(f"原始预处理器: {original_time:.4f}秒 (1000次)")
    
    start_time = time.time()
    for text in test_texts * 100:
        enhanced_preprocessor.preprocess(text)
    enhanced_time = time.time() - start_time
    print(f"增强预处理器: {enhanced_time:.4f}秒 (1000次)")
    print()

if __name__ == "__main__":
    test_preprocessors()