# 文本预处理模块测试

import unittest
import sys
import os

# 添加training_new目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """测试文本预处理模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """测试文本清洗功能"""
        test_text = "这是一段测试文本，包含特殊字符！@#$%^&*()"
        cleaned_text = self.preprocessor.clean_text(test_text)
        self.assertEqual(cleaned_text, "这是一段测试文本包含特殊字符")
        self.assertNotIn("!", cleaned_text)
        self.assertNotIn("@", cleaned_text)
    
    def test_tokenize(self):
        """测试分词功能"""
        test_text = "这是一段测试文本"
        tokens = self.preprocessor.tokenize(test_text)
        self.assertIsInstance(tokens, list)
        self.assertIn("这是", tokens)
        self.assertIn("一段", tokens)
        self.assertIn("测试", tokens)
        self.assertIn("文本", tokens)
    
    def test_remove_duplicate_phrases(self):
        """测试去重复短语功能"""
        test_words = ["测试", "测试", "文本", "文本", "重复", "重复"]
        unique_words = self.preprocessor.remove_duplicate_phrases(test_words)
        self.assertIsInstance(unique_words, list)
        self.assertEqual(len(unique_words), 3)
        self.assertIn("测试", unique_words)
        self.assertIn("文本", unique_words)
        self.assertIn("重复", unique_words)
    
    def test_is_repetitive_interjection(self):
        """测试是否为重复语气词"""
        # 测试重复语气词
        self.assertTrue(self.preprocessor._is_repetitive_interjection("哈哈哈哈"))
        self.assertTrue(self.preprocessor._is_repetitive_interjection("啊啊啊啊"))
        
        # 测试非重复语气词
        self.assertFalse(self.preprocessor._is_repetitive_interjection("哈哈"))
        self.assertFalse(self.preprocessor._is_repetitive_interjection("测试"))
    
    def test_remove_stopwords(self):
        """测试去停用词功能"""
        test_words = ["的", "是", "在", "测试", "文本", "哈哈哈哈"]
        filtered_words = self.preprocessor.remove_stopwords(test_words)
        self.assertIsInstance(filtered_words, list)
        self.assertNotIn("的", filtered_words)
        self.assertNotIn("是", filtered_words)
        self.assertNotIn("在", filtered_words)
        self.assertNotIn("哈哈哈哈", filtered_words)
        self.assertIn("测试", filtered_words)
        self.assertIn("文本", filtered_words)
    
    def test_preprocess(self):
        """测试完整预处理流程"""
        test_text = "这是一段测试文本，包含停用词和重复语气词哈哈哈哈"
        processed_words = self.preprocessor.preprocess(test_text)
        self.assertIsInstance(processed_words, list)
        self.assertIn("测试", processed_words)
        self.assertIn("文本", processed_words)
        self.assertNotIn("的", processed_words)
        self.assertNotIn("是", processed_words)
        self.assertNotIn("哈哈哈哈", processed_words)


if __name__ == '__main__':
    unittest.main()