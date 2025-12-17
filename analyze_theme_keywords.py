#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

def load_theme_keywords():
    """加载主题关键词"""
    try:
        keywords_path = os.path.join('data', 'models', 'theme_keywords.pkl')
        with open(keywords_path, 'rb') as f:
            theme_keywords = pickle.load(f)
        
        print("主题关键词分析：")
        print("=" * 50)
        
        for theme, keywords in theme_keywords.items():
            print(f"\n{theme}:")
            print(f"关键词: {', '.join(keywords[:10])}")  # 显示前10个关键词
            
            # 尝试根据关键词推断主题名称
            theme_name = generate_theme_name(theme, keywords)
            print(f"建议名称: {theme_name}")
        
        return theme_keywords
        
    except Exception as e:
        print(f"加载主题关键词失败: {e}")
        return None

def generate_theme_name(theme_id, keywords):
    """根据关键词生成主题名称"""
    keyword_str = ' '.join(keywords)
    
    # 根据关键词特征推断主题
    if any(word in keyword_str for word in ['软件', '会员', '交友', '社交']):
        return "社交软件应用"
    elif any(word in keyword_str for word in ['下载', '对象', '脱单', 'boss']):
        return "婚恋求职服务"
    elif any(word in keyword_str for word in ['聊天', '照片', '搭子', '交流']):
        return "聊天交友互动"
    elif any(word in keyword_str for word in ['soul', '感觉', 'app', '体验']):
        return "APP体验分享"
    elif any(word in keyword_str for word in ['好玩', '喜欢', '认识', '免费']):
        return "娱乐休闲推荐"
    else:
        return f"主题{theme_id.split(' ')[-1]}"

if __name__ == "__main__":
    load_theme_keywords()