"""
模型管理器
统一管理LDA模型、主题分类器和相关工具
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from src.config import config
from src.logger import Logger, log_function_call, log_error, log_info
from src.text_preprocessor import get_preprocessor

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.logger = Logger.get_logger(__name__)
        self.lda_model = None
        self.theme_classifier = None
        self.vectorizer = None
        self.count_vectorizer = None
        self.theme_keywords = None
        self.preprocessor = get_preprocessor()
        
        # 加载所有模型
        self._load_models()
    
    @log_function_call
    def _load_models(self):
        """加载所有模型文件"""
        model_status = {}
        
        # 加载LDA模型
        try:
            if config['default'].MODEL_FILES['lda_model'].exists():
                self.lda_model = joblib.load(config['default'].MODEL_FILES['lda_model'])
                model_status['lda_model'] = '成功'
            else:
                model_status['lda_model'] = '文件不存在'
        except Exception as e:
            model_status['lda_model'] = f'加载失败: {str(e)}'
            log_error(e, "LDA模型加载失败")
        
        # 加载主题分类器
        try:
            if config['default'].MODEL_FILES['theme_classifier'].exists():
                self.theme_classifier = joblib.load(config['default'].MODEL_FILES['theme_classifier'])
                model_status['theme_classifier'] = '成功'
            else:
                model_status['theme_classifier'] = '文件不存在'
        except Exception as e:
            model_status['theme_classifier'] = f'加载失败: {str(e)}'
            log_error(e, "主题分类器加载失败")
        
        # 加载向量化器
        try:
            if config['default'].MODEL_FILES['vectorizer'].exists():
                self.vectorizer = joblib.load(config['default'].MODEL_FILES['vectorizer'])
                model_status['vectorizer'] = '成功'
            else:
                model_status['vectorizer'] = '文件不存在'
        except Exception as e:
            model_status['vectorizer'] = f'加载失败: {str(e)}'
            log_error(e, "向量化器加载失败")
        
        # 加载Count向量化器
        try:
            if config['default'].MODEL_FILES['count_vectorizer'].exists():
                self.count_vectorizer = joblib.load(config['default'].MODEL_FILES['count_vectorizer'])
                model_status['count_vectorizer'] = '成功'
            else:
                model_status['count_vectorizer'] = '文件不存在'
        except Exception as e:
            model_status['count_vectorizer'] = f'加载失败: {str(e)}'
            log_error(e, "Count向量化器加载失败")
        
        # 加载主题关键词
        try:
            if config['default'].MODEL_FILES['theme_keywords'].exists():
                self.theme_keywords = joblib.load(config['default'].MODEL_FILES['theme_keywords'])
                model_status['theme_keywords'] = '成功'
            else:
                model_status['theme_keywords'] = '文件不存在'
                # 创建默认主题关键词
                self.theme_keywords = self._create_default_theme_keywords()
        except Exception as e:
            model_status['theme_keywords'] = f'加载失败: {str(e)}'
            log_error(e, "主题关键词加载失败")
            self.theme_keywords = self._create_default_theme_keywords()
        
        log_info(f"模型加载完成: {model_status}")
    
    @log_function_call
    def _create_default_theme_keywords(self) -> Dict[str, List[str]]:
        """创建默认主题关键词"""
        default_keywords = {}
        for theme in config['default'].THEME_CONFIG['default_themes']:
            default_keywords[theme] = [f"关键词{i+1}" for i in range(5)]
        
        log_info("使用默认主题关键词")
        return default_keywords
    
    @log_function_call
    def preprocess_text(self, text: str) -> List[str]:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的词语列表
        """
        return self.preprocessor.preprocess(text)
    
    @log_function_call
    def extract_lda_features(self, processed_text: List[str]) -> Optional[np.ndarray]:
        """
        提取LDA特征
        
        Args:
            processed_text: 预处理后的文本
            
        Returns:
            LDA特征向量
        """
        try:
            if self.count_vectorizer is None or self.lda_model is None:
                log_error("LDA模型或向量化器未加载", "LDA特征提取失败")
                return None
            
            # 将词语列表转换为字符串
            text_str = ' '.join(processed_text)
            
            # 向量化
            text_vector = self.count_vectorizer.transform([text_str])
            
            # LDA转换
            lda_features = self.lda_model.transform(text_vector)
            
            return lda_features[0]
            
        except Exception as e:
            log_error(e, "LDA特征提取失败")
            return None
    
    @log_function_call
    def extract_tfidf_features(self, processed_text: List[str]) -> Optional[np.ndarray]:
        """
        提取TF-IDF特征
        
        Args:
            processed_text: 预处理后的文本
            
        Returns:
            TF-IDF特征向量
        """
        try:
            if self.vectorizer is None:
                log_error("TF-IDF向量化器未加载", "TF-IDF特征提取失败")
                return None
            
            # 将词语列表转换为字符串
            text_str = ' '.join(processed_text)
            
            # TF-IDF向量化
            tfidf_features = self.vectorizer.transform([text_str])
            
            return tfidf_features.toarray()[0]
            
        except Exception as e:
            log_error(e, "TF-IDF特征提取失败")
            return None
    
    @log_function_call
    def predict_theme(self, text: str) -> Dict[str, Any]:
        """
        预测文本主题
        
        Args:
            text: 原始文本
            
        Returns:
            预测结果字典
        """
        try:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {
                    'success': False,
                    'error': '文本预处理失败或结果为空',
                    'theme': None,
                    'confidence': 0.0
                }
            
            # 提取特征
            lda_features = self.extract_lda_features(processed_text)
            tfidf_features = self.extract_tfidf_features(processed_text)
            
            if lda_features is None or tfidf_features is None:
                return {
                    'success': False,
                    'error': '特征提取失败',
                    'theme': None,
                    'confidence': 0.0
                }
            
            # 合并特征
            combined_features = np.concatenate([lda_features, tfidf_features])
            
            # 预测主题
            if self.theme_classifier is None:
                return {
                    'success': False,
                    'error': '主题分类器未加载',
                    'theme': None,
                    'confidence': 0.0
                }
            
            # 获取预测概率
            probabilities = self.theme_classifier.predict_proba([combined_features])[0]
            predicted_class = self.theme_classifier.predict([combined_features])[0]
            
            # 获取主题名称
            theme_names = list(self.theme_keywords.keys()) if self.theme_keywords else THEME_CONFIG['default_themes']
            
            if predicted_class < len(theme_names):
                predicted_theme = theme_names[predicted_class]
                confidence = float(probabilities[predicted_class])
            else:
                predicted_theme = "未知主题"
                confidence = 0.0
            
            return {
                'success': True,
                'theme': predicted_theme,
                'confidence': confidence,
                'processed_text': processed_text,
                'probabilities': {
                    theme_names[i] if i < len(theme_names) else f"主题{i}": float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            log_error(e, "主题预测失败")
            return {
                'success': False,
                'error': f'预测失败: {str(e)}',
                'theme': None,
                'confidence': 0.0
            }
    
    @log_function_call
    def get_theme_keywords(self, theme: str = None) -> Dict[str, List[str]]:
        """
        获取主题关键词
        
        Args:
            theme: 指定主题，如果为None则返回所有主题关键词
            
        Returns:
            主题关键词字典
        """
        if self.theme_keywords is None:
            return {}
        
        if theme is None:
            return self.theme_keywords
        else:
            return {theme: self.theme_keywords.get(theme, [])}
    
    @log_function_call
    def get_theme_name(self, theme: str) -> str:
        """
        获取主题的友好名称 - 基于独特关键词特征进行精确区分
        
        Args:
            theme: 主题标识符（如"主题 1"）
            
        Returns:
            主题的友好名称
        """
        if self.theme_keywords is None or theme not in self.theme_keywords:
            return theme
        
        keywords = self.theme_keywords[theme]
        keyword_str = ' '.join(keywords)
        
        # 基于实际关键词特征的精确命名逻辑
        if theme == "主题 1":
            # 主题1关键词: ['软件', '会员', '可以', '厉害', '一个', '交友', '这些', '这个', '各取所需', '没有']
            if '会员' in keyword_str and '交友' in keyword_str:
                return "会员制社交软件"
            elif '软件' in keyword_str and '厉害' in keyword_str:
                return "软件功能评价"
            else:
                return "社交软件应用"
        
        elif theme == "主题 2":
            # 主题2关键词: ['下载', '第一', '对象', '美团', 'boss', '脱单', '为什', '可以', '会员', '很多']
            if 'boss' in keyword_str or '美团' in keyword_str:
                return "求职招聘平台"
            elif '脱单' in keyword_str or '对象' in keyword_str:
                return "婚恋脱单服务"
            else:
                return "婚恋求职服务"
        
        elif theme == "主题 3":
            # 主题3关键词: ['这个', '聊天', '照片', '搭子', '知道', '还是', '结果', '他们', '对象', '可以']
            if '搭子' in keyword_str or '聊天' in keyword_str:
                return "社交搭子交友"
            elif '照片' in keyword_str:
                return "照片社交互动"
            else:
                return "婚恋交友服务"
        
        elif theme == "主题 4":
            # 主题4关键词: ['一个', '可以', 'soul', '感觉', 'app', '现在', '这个', '没有', '时候', '上面']
            if 'soul' in keyword_str:
                return "Soul APP体验"
            elif '感觉' in keyword_str or 'app' in keyword_str:
                return "APP使用感受"
            else:
                return "APP体验分享"
        
        elif theme == "主题 5":
            # 主题5关键词: ['软件', '现在', '好玩', '喜欢', '一个', '认识', '自己', '免费', '这个', '没有']
            if '好玩' in keyword_str or '喜欢' in keyword_str:
                return "娱乐社交软件"
            elif '免费' in keyword_str and '认识' in keyword_str:
                return "免费交友平台"
            elif '软件' in keyword_str:
                return "社交软件推荐"
            else:
                return "软件娱乐体验"
        
        # 通用命名逻辑（用于其他主题）
        elif any(word in keyword_str for word in ['boss', '招聘', '求职', '工作', '面试']):
            return "婚恋求职服务"
        elif any(word in keyword_str for word in ['脱单', '对象', '相亲', '恋爱', '结婚']):
            return "婚恋交友服务"
        elif any(word in keyword_str for word in ['下载', '安装', '软件', 'app', '应用']):
            return "APP下载讨论"
        elif any(word in keyword_str for word in ['soul', '体验', '感觉', '使用', '功能']):
            return "APP体验分享"
        elif any(word in keyword_str for word in ['聊天', '搭子', '交流', '互动', '认识']):
            return "社交互动交流"
        elif any(word in keyword_str for word in ['会员', '付费', '价格', '费用', 'vip']):
            return "会员服务讨论"
        elif any(word in keyword_str for word in ['好玩', '喜欢', '推荐', '免费', '娱乐']):
            return "娱乐休闲推荐"
        elif any(word in keyword_str for word in ['游戏', '玩', '装备', '等级']):
            return "游戏相关讨论"
        else:
            # 提取最常见的关键词作为名称
            if keywords:
                # 过滤掉通用词
                filtered_keywords = [kw for kw in keywords if kw not in ['可以', '一个', '这个', '没有', '现在', '时候']]
                if filtered_keywords:
                    return f"{filtered_keywords[0]}相关讨论"
            return theme
    
    @log_function_call
    def get_all_theme_names(self) -> Dict[str, str]:
        """
        获取所有主题的友好名称
        
        Returns:
            主题标识符到友好名称的映射
        """
        if self.theme_keywords is None:
            return {}
        
        theme_names = {}
        for theme in self.theme_keywords.keys():
            theme_names[theme] = self.get_theme_name(theme)
        
        return theme_names
    
    @log_function_call
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量分析文本
        
        Args:
            texts: 文本列表
            
        Returns:
            分析结果列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict_theme(text)
                result['text_index'] = i
                result['original_text'] = text[:100] + "..." if len(text) > 100 else text
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    log_info(f"已分析 {i + 1}/{len(texts)} 条文本")
                    
            except Exception as e:
                log_error(e, f"分析第{i+1}条文本失败")
                results.append({
                    'text_index': i,
                    'original_text': text[:100] + "..." if len(text) > 100 else text,
                    'success': False,
                    'error': str(e),
                    'theme': None,
                    'confidence': 0.0
                })
        
        return results
    
    def get_model_status(self) -> Dict[str, str]:
        """获取模型状态"""
        return {
            'lda_model': '已加载' if self.lda_model is not None else '未加载',
            'theme_classifier': '已加载' if self.theme_classifier is not None else '未加载',
            'vectorizer': '已加载' if self.vectorizer is not None else '未加载',
            'count_vectorizer': '已加载' if self.count_vectorizer is not None else '未加载',
            'theme_keywords': '已加载' if self.theme_keywords is not None else '未加载',
        }

# 全局模型管理器实例
_model_manager = None

def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

if __name__ == '__main__':
    # 测试模型管理器
    manager = ModelManager()
    
    # 显示模型状态
    status = manager.get_model_status()
    print("模型状态:")
    for model, state in status.items():
        print(f"  {model}: {state}")
    
    # 测试预测
    test_text = "今天去了一家很棒的餐厅，吃了美味的意大利面和甜点。"
    result = manager.predict_theme(test_text)
    print(f"\n测试文本: {test_text}")
    print(f"预测结果: {result}")