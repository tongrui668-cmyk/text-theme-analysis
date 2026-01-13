# 模型保存模块

import os
import joblib
from src.config import config


class ModelSaver:
    """模型保存类"""
    
    def __init__(self):
        """初始化模型保存器"""
        pass
    
    def save_models(self, models):
        """
        保存模型和向量化器
        
        Args:
            models: 模型字典，包含所有需要保存的模型
            
        Returns:
            dict: 保存的模型路径
        """
        # 创建模型目录
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        print(f"\n保存模型到 {config.MODEL_DIR}...")
        
        # 保存模型
        model_paths = {}
        
        if 'lda' in models:
            lda_path = os.path.join(config.MODEL_DIR, 'lda_model.pkl')
            joblib.dump(models['lda'], lda_path)
            model_paths['lda'] = lda_path
        
        if 'count_vectorizer' in models:
            cv_path = os.path.join(config.MODEL_DIR, 'count_vectorizer.pkl')
            joblib.dump(models['count_vectorizer'], cv_path)
            model_paths['count_vectorizer'] = cv_path
        
        if 'tfidf_vectorizer' in models:
            tfidf_path = os.path.join(config.MODEL_DIR, 'vectorizer.pkl')
            joblib.dump(models['tfidf_vectorizer'], tfidf_path)
            model_paths['tfidf_vectorizer'] = tfidf_path
        
        if 'classifier' in models:
            clf_path = os.path.join(config.MODEL_DIR, 'theme_classification_model.pkl')
            joblib.dump(models['classifier'], clf_path)
            model_paths['classifier'] = clf_path
        
        if 'keywords' in models:
            keywords_path = os.path.join(config.MODEL_DIR, 'theme_keywords.pkl')
            joblib.dump(models['keywords'], keywords_path)
            model_paths['keywords'] = keywords_path
        
        print("\n所有模型已保存完成！")
        for name, path in model_paths.items():
            print(f"- {name}: {os.path.basename(path)}")
        
        return model_paths
