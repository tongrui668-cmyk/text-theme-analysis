#!/usr/bin/env python3
# 主训练脚本

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.data_preprocessor import DataPreprocessor
from src.topic_modeler import TopicModeler
from src.label_generator import LabelGenerator
from src.keyword_extractor import KeywordExtractor
from src.classifier import ThemeClassifier
from src.model_saver import ModelSaver
from src.report_generator import ReportGenerator
from src.text_preprocessor_enhanced import EnhancedTextPreprocessor


def main():
    """主训练函数"""
    
    # 1. 初始化组件
    print("初始化训练组件...")
    text_preprocessor = EnhancedTextPreprocessor()
    data_preprocessor = DataPreprocessor(text_preprocessor)
    topic_modeler = TopicModeler()
    label_generator = LabelGenerator()
    keyword_extractor = KeywordExtractor()
    classifier = ThemeClassifier()
    model_saver = ModelSaver()
    report_generator = ReportGenerator()
    
    # 2. 加载和预处理数据
    df = data_preprocessor.load_data()
    df = data_preprocessor.preprocess_data(df)
    df_train, df_val, df_test = data_preprocessor.split_data(df)
    
    # 3. 向量化文本数据
    print("开始向量化文本数据（仅使用训练集）...")
    vectorizer = topic_modeler.create_count_vectorizer()
    X_train_count = vectorizer.fit_transform(df_train['cleaned_text'])
    print(f"向量化完成，特征维度: {X_train_count.shape[1]}")
    
    # 4. 评估最佳主题数
    n_topics, best_perplexity, evaluation_results = topic_modeler.evaluate_topic_numbers(X_train_count)
    
    # 5. 训练LDA模型
    lda = topic_modeler.train_lda(X_train_count, n_topics)
    
    # 6. 生成业务标签
    print("基于关键词自动生成业务标签...")
    business_labels = label_generator.get_business_labels(n_topics, lda, vectorizer)
    print("业务标签生成完成")
    
    # 7. 为训练集分配主题
    print("为训练集分配主题...")
    doc_topic_dist_train = topic_modeler.get_topic_distribution(X_train_count)
    df_train['dominant_topic'] = doc_topic_dist_train.argmax(axis=1)
    
    # 8. 准备TF-IDF特征
    print("创建TF-IDF向量化器（仅使用训练集）...")
    tfidf_vectorizer = topic_modeler.create_tfidf_vectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['cleaned_text'])
    print(f"TF-IDF向量化完成，特征维度: {X_train_tfidf.shape[1]}")
    
    # 9. 提取关键词
    lda_keywords = keyword_extractor.get_lda_keywords(lda, vectorizer, business_labels)
    print("\n自动生成的主题关键词：")
    for theme, words in lda_keywords.items():
        print(f"{theme}: {', '.join(words[:10])}")
    
    tfidf_keywords = keyword_extractor.get_tfidf_keywords_by_topic(df_train, tfidf_vectorizer, business_labels)
    print("\n提取每个主题的TF-IDF关键词：")
    for theme, words in tfidf_keywords.items():
        print(f"{theme} (TF-IDF): {', '.join(words[:10])}")
    
    combined_keywords = keyword_extractor.get_combined_keywords(lda_keywords, tfidf_keywords)
    print("\n结合LDA和TF-IDF关键词：")
    for theme, words in combined_keywords.items():
        print(f"{theme} (组合): {', '.join(words[:10])}")
    
    # 10. 合并特征
    print("为训练集合并特征...")
    import numpy as np
    X_combined_train = np.hstack([doc_topic_dist_train, X_train_tfidf.toarray()])
    print(f"合并后特征维度: {X_combined_train.shape[1]}")
    
    # 11. 为验证集提取特征
    print("为验证集提取特征...")
    X_val_count = vectorizer.transform(df_val['cleaned_text'])
    X_val_tfidf = tfidf_vectorizer.transform(df_val['cleaned_text'])
    doc_topic_dist_val = topic_modeler.get_topic_distribution(X_val_count)
    X_combined_val = np.hstack([doc_topic_dist_val, X_val_tfidf.toarray()])
    df_val['dominant_topic'] = doc_topic_dist_val.argmax(axis=1)
    
    # 12. 为测试集提取特征
    print("为测试集提取特征...")
    X_test_count = vectorizer.transform(df_test['cleaned_text'])
    X_test_tfidf = tfidf_vectorizer.transform(df_test['cleaned_text'])
    doc_topic_dist_test = topic_modeler.get_topic_distribution(X_test_count)
    X_combined_test = np.hstack([doc_topic_dist_test, X_test_tfidf.toarray()])
    df_test['dominant_topic'] = doc_topic_dist_test.argmax(axis=1)
    
    # 13. 训练分类器
    classifier.create_classifier()
    classifier.train(X_combined_train, df_train['dominant_topic'])
    
    # 14. 评估模型
    # 验证集评估
    val_accuracy = classifier.evaluate(X_combined_val, df_val['dominant_topic'], business_labels)['accuracy']
    # 测试集评估
    test_evaluation = classifier.evaluate(X_combined_test, df_test['dominant_topic'], business_labels)
    test_accuracy = test_evaluation['accuracy']
    
    # 计算困惑度
    perplexity = lda.perplexity(X_train_count)
    print(f"\nLDA模型困惑度: {perplexity:.2f}")
    
    # 15. 收集训练结果
    training_results = {
        'total_samples': len(df),
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_test),
        'feature_dim': X_train_count.shape[1],
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'perplexity': perplexity,
        'n_topics': n_topics,
        'evaluation_results': evaluation_results,
        'business_labels': business_labels,
        'lda_model': lda,
        'vectorizer': vectorizer,
        'tfidf_keywords': tfidf_keywords,
        'combined_keywords': combined_keywords
    }
    
    # 16. 生成训练报告
    report_path = report_generator.generate_report(training_results)
    
    # 17. 保存模型
    models_to_save = {
        'lda': lda,
        'count_vectorizer': vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer,
        'classifier': classifier.classifier,
        'keywords': combined_keywords
    }
    model_paths = model_saver.save_models(models_to_save)
    
    # 18. 打印最终结果
    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)
    print(f"最佳主题数: {n_topics}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"LDA困惑度: {perplexity:.2f}")
    print(f"训练报告: {report_path}")
    print(f"模型保存路径: {config.MODEL_DIR}")
    print("\n主题业务标签：")
    for theme, label in business_labels.items():
        print(f"{theme}: {label}")


if __name__ == "__main__":
    main()
