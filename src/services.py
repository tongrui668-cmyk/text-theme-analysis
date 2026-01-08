"""
业务逻辑服务模块
将文件处理和文本分析的业务逻辑分离
"""
import os
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import current_app, flash, redirect, url_for, render_template

from src.logger import log_info, log_error
from src.model_manager import get_model_manager


class FileProcessingService:
    """文件处理服务"""
    
    def __init__(self):
        self.model_manager = None
        self._init_model_manager()
    
    def _init_model_manager(self):
        """初始化模型管理器"""
        try:
            self.model_manager = get_model_manager()
        except Exception as e:
            log_error(e, "模型管理器初始化失败")
    
    def allowed_file(self, filename):
        """检查文件扩展名是否允许"""
        if not filename:
            return False, "文件名不能为空"
        
        if '.' not in filename:
            return False, "文件缺少扩展名"
        
        extension = filename.rsplit('.', 1)[1].lower()
        allowed_extensions = {'xlsx', 'xls', 'txt', 'csv'}
        
        if extension not in allowed_extensions:
            return False, f"不支持的文件格式：.{extension}。支持的格式：.txt, .xlsx, .xls, .csv"
        
        return True, "文件格式验证通过"
    
    def validate_file_size(self, file):
        """验证文件大小"""
        # 重置文件指针到开始位置
        file.seek(0, 2)  # 移动到文件末尾
        size = file.tell()  # 获取文件大小
        file.seek(0)  # 重置到开始位置
        
        max_size = 10 * 1024 * 1024  # 10MB
        if size > max_size:
            return False, f"文件大小 {size/1024/1024:.1f}MB 超过限制 (最大10MB)"
        
        if size == 0:
            return False, "文件为空"
        
        return True, f"文件大小 {size/1024:.1f}KB，验证通过"
    
    def save_uploaded_file(self, file, filename):
        """保存上传的文件"""
        upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
        upload_folder.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_folder / filename
        file.save(str(filepath))
        return filepath
    
    def process_excel_file(self, filepath):
        """处理Excel文件"""
        if self.model_manager is None:
            # 跳转到结果界面显示错误
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': '模型管理器未初始化，请重启应用',
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        log_info(f"开始读取Excel文件: {filepath}, 大小: {os.path.getsize(filepath)/1024/1024:.2f}MB")
        
        # 预处理检查
        if not os.path.exists(filepath):
            error_msg = 'Excel文件处理失败: 文件不存在\n\n可能的解决方法：\n'
            error_msg += '1. 确保文件上传成功\n'
            error_msg += '2. 检查文件路径是否正确\n'
            error_msg += '3. 尝试重新上传文件'
            
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': error_msg,
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        # 检查文件大小
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            error_msg = 'Excel文件处理失败: 文件为空\n\n可能的解决方法：\n'
            error_msg += '1. 确保上传的文件包含数据\n'
            error_msg += '2. 检查文件是否正确保存\n'
            error_msg += '3. 尝试重新创建或导出Excel文件'
            
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': error_msg,
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        # 读取Excel文件 - 尝试多种方法
        log_info("读取Excel文件...")
        df = None
        
        # 方法1: 尝试使用openpyxl引擎
        try:
            log_info("尝试使用openpyxl引擎读取...")
            df = pd.read_excel(
                filepath,
                engine='openpyxl',
                keep_default_na=False,
                dtype=str,
                header=0
            )
            log_info("使用openpyxl引擎读取成功")
        except Exception as e1:
            log_error(e1, "使用openpyxl引擎读取失败")
            
            # 方法2: 尝试使用xlrd引擎（如果可用）
            try:
                log_info("尝试使用xlrd引擎读取...")
                df = pd.read_excel(
                    filepath,
                    engine='xlrd',
                    keep_default_na=False,
                    dtype=str,
                    header=0
                )
                log_info("使用xlrd引擎读取成功")
            except Exception as e2:
                log_error(e2, "使用xlrd引擎读取失败")
                
                # 方法3: 尝试使用默认引擎
                try:
                    log_info("尝试使用默认引擎读取...")
                    df = pd.read_excel(
                        filepath,
                        keep_default_na=False,
                        dtype=str,
                        header=0
                    )
                    log_info("使用默认引擎读取成功")
                except Exception as e3:
                    log_error(e3, "所有引擎读取失败")
                    error_msg = f'Excel文件处理失败: {str(e3)}\n\n可能的解决方法：\n'
                    error_msg += '1. 确保Excel文件格式完整，没有损坏\n'
                    error_msg += '2. 检查文件扩展名是否与实际格式匹配（不要将CSV文件重命名为.xlsx）\n'
                    error_msg += '3. 确认文件已正确保存并包含数据\n'
                    error_msg += '4. 尝试重新导出或创建Excel文件\n'
                    error_msg += '5. 检查文件大小是否超过限制（当前限制：10MB）'
                    
                    result_data = {
                        'filename': filepath.name,
                        'total_texts': 0,
                        'successful_analyses': 0,
                        'error_message': error_msg,
                        'error': True
                    }
                    return render_template('result.html', result=result_data)
        
        log_info(f"Excel文件读取完成，行数: {len(df)}, 列数: {len(df.columns)}")
        
        # 检查是否为空数据框
        if df.empty:
            error_msg = 'Excel文件处理失败: 文件中没有数据\n\n可能的解决方法：\n'
            error_msg += '1. 确保Excel文件包含数据\n'
            error_msg += '2. 检查数据是否在第一个工作表中\n'
            error_msg += '3. 尝试重新导出Excel文件，确保数据正确保存'
            
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': error_msg,
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        # 查找评论列
        text_column = self._find_text_column(df)
        
        if text_column is None:
            # 跳转到结果界面显示错误
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': '文件中未找到包含"评论"的列。本系统专门用于分析评论数据，请确保上传的文件包含评论列。',
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        # 获取文本数据
        texts = self._extract_texts_from_dataframe(df, text_column)
        
        if not texts:
            # 跳转到结果界面显示错误
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': '评论列中没有找到有效的文本内容',
                'error': True
            }
            return render_template('result.html', result=result_data)
        
        # 批量分析
        results = self.model_manager.analyze_batch(texts)
        
        # 统计结果
        result_data = self._analyze_results(results, filepath.name, len(texts))
        
        log_info(f"Excel文件处理完成: {filepath.name}, 成功分析 {result_data['successful_analyses']}/{len(texts)} 条文本")
        
        return render_template('result.html', result=result_data)
    
    def process_text_file(self, filepath):
        """处理文本文件"""
        try:
            if self.model_manager is None:
                # 跳转到结果界面显示错误
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 0,
                    'successful_analyses': 0,
                    'error_message': '模型管理器未初始化，请重启应用',
                    'error': True
                }
                return render_template('result.html', result=result_data)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分析文本
            result = self.model_manager.predict_theme(content)
            
            if result.get('success', False):
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 1,
                    'successful_analyses': 1,
                    'single_result': result
                }
                
                log_info(f"文本文件处理完成: {filepath.name}")
                return render_template('result.html', result=result_data)
            else:
                # 跳转到结果界面显示错误
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 1,
                    'successful_analyses': 0,
                    'error_message': f'文本分析失败: {result.get("error", "未知错误")}',
                    'error': True
                }
                return render_template('result.html', result=result_data)
                
        except Exception as e:
            log_error(e, f"文本文件处理失败: {filepath}")
            # 跳转到结果界面显示错误
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': f'文本文件处理失败: {str(e)}',
                'error': True
            }
            return render_template('result.html', result=result_data)
    
    def process_csv_file(self, filepath):
        """处理CSV文件"""
        try:
            if self.model_manager is None:
                # 跳转到结果界面显示错误
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 0,
                    'successful_analyses': 0,
                    'error_message': '模型管理器未初始化，请重启应用',
                    'error': True
                }
                return render_template('result.html', result=result_data)
            
            # 读取CSV文件
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # 查找评论列
            text_column = self._find_text_column(df)
            
            if text_column is None:
                # 跳转到结果界面显示错误
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 0,
                    'successful_analyses': 0,
                    'error_message': '文件中未找到包含"评论"的列。本系统专门用于分析评论数据，请确保上传的文件包含评论列。',
                    'error': True
                }
                return render_template('result.html', result=result_data)
            
            # 获取文本数据
            texts = self._extract_texts_from_dataframe(df, text_column)
            
            if not texts:
                # 跳转到结果界面显示错误
                result_data = {
                    'filename': filepath.name,
                    'total_texts': 0,
                    'successful_analyses': 0,
                    'error_message': '评论列中没有找到有效的文本内容',
                    'error': True
                }
                return render_template('result.html', result=result_data)
            
            # 批量分析
            results = self.model_manager.analyze_batch(texts)
            
            # 统计结果
            result_data = self._analyze_results(results, filepath.name, len(texts))
            
            log_info(f"CSV文件处理完成: {filepath.name}, 成功分析 {result_data['successful_analyses']}/{len(texts)} 条文本")
            
            return render_template('result.html', result=result_data)
            
        except Exception as e:
            log_error(e, f"CSV文件处理失败: {filepath}")
            # 跳转到结果界面显示错误
            result_data = {
                'filename': filepath.name,
                'total_texts': 0,
                'successful_analyses': 0,
                'error_message': f'CSV文件处理失败: {str(e)}',
                'error': True
            }
            return render_template('result.html', result=result_data)
    
    def _find_text_column(self, df):
        """查找文本列 - 专门用于评论内容数据"""
        # 优先级1: 查找包含"评论内容"的列（最精确）
        for col in df.columns:
            if '评论内容' in col:
                return col
        
        # 优先级2: 查找同时包含"评论"和"内容"的列
        for col in df.columns:
            if '评论' in col and '内容' in col:
                return col
        
        # 优先级3: 查找包含"内容"的列
        for col in df.columns:
            if '内容' in col:
                return col
        
        # 优先级4: 查找包含"评论"但排除明显不是内容的列
        content_keywords = ['内容', '正文', '文本', '留言', '评论文本']
        exclude_keywords = ['数', '量', '人', '时间', '日期', 'ip', 'ID', 'id']
        
        for col in df.columns:
            if '评论' in col:
                # 检查是否包含排除关键词
                if not any(keyword in col for keyword in exclude_keywords):
                    return col
                # 检查是否包含内容关键词
                if any(keyword in col for keyword in content_keywords):
                    return col
        
        # 如果没有找到合适的评论内容列，返回None表示不符合要求
        return None
    
    def _extract_texts_from_dataframe(self, df, text_column):
        """从DataFrame中提取文本数据，过滤掉过短的文本和重复内容"""
        # 获取文本数据并填充空值
        texts = df[text_column].fillna('').astype(str)
        
        # 过滤掉长度<=5的文本（这些文本通常没有足够的语义信息）
        filtered_texts = [text for text in texts if len(text.strip()) > 5]
        
        # 去重处理 - 移除重复的评论内容
        unique_texts = []
        seen_texts = set()
        duplicate_count = 0
        
        for text in filtered_texts:
            # 使用文本的哈希值来检测重复
            text_hash = hash(text.strip())
            if text_hash not in seen_texts:
                unique_texts.append(text)
                seen_texts.add(text_hash)
            else:
                duplicate_count += 1
        
        log_info(f"文本提取完成: 原始 {len(texts)} 条，长度过滤后 {len(filtered_texts)} 条，去重后 {len(unique_texts)} 条")
        log_info(f"去重统计: 移除 {duplicate_count} 条重复评论，去重率 {(duplicate_count/len(filtered_texts)*100):.1f}%")
        
        return unique_texts
    
    def _analyze_results(self, results, filename, total_texts):
        """分析结果并统计数据"""
        themes_count = {}
        successful_analyses = 0
        theme_examples = {}  # 存储每个主题的示例评论
        theme_keywords = {}  # 存储每个主题的关键词
        
        for result in results:
            if result.get('success', False):
                successful_analyses += 1
                theme = result.get('theme', '未知')
                themes_count[theme] = themes_count.get(theme, 0) + 1
                
                # 收集每个主题的示例评论（最多5个）
                if theme not in theme_examples:
                    theme_examples[theme] = []
                if len(theme_examples[theme]) < 5:
                    theme_examples[theme].append({
                        'text': result.get('original_text', ''),
                        'confidence': result.get('confidence', 0.0),
                        'keywords': result.get('keywords', [])
                    })
                
                # 收集主题关键词
                if theme not in theme_keywords and result.get('keywords'):
                    theme_keywords[theme] = result.get('keywords', [])[:5]
        
        # 获取主题友好名称
        theme_names = {}
        if self.model_manager:
            theme_names = self.model_manager.get_all_theme_names()
        
        return {
            'filename': filename,
            'total_texts': total_texts,
            'successful_analyses': successful_analyses,
            'themes_count': themes_count,
            'theme_names': theme_names,
            'theme_examples': theme_examples,
            'theme_keywords': theme_keywords,
            'results': results[:10],  # 只显示前10个结果
            'show_all': len(results) > 10
        }


class TextAnalysisService:
    """文本分析服务"""
    
    def __init__(self):
        self.model_manager = None
        self._init_model_manager()
    
    def _init_model_manager(self):
        """初始化模型管理器"""
        try:
            self.model_manager = get_model_manager()
        except Exception as e:
            log_error(e, "模型管理器初始化失败")
    
    def analyze_text(self, text):
        """分析单个文本"""
        try:
            if self.model_manager is None:
                return {
                    'success': False,
                    'error': '模型管理器未初始化'
                }
            
            return self.model_manager.predict_theme(text)
            
        except Exception as e:
            log_error(e, "文本分析失败")
            return {
                'success': False,
                'error': f'分析失败: {str(e)}'
            }
    
    def get_service_status(self):
        """获取服务状态"""
        try:
            if self.model_manager is None:
                return {
                    'success': False,
                    'error': '模型管理器未初始化'
                }
            
            model_status = self.model_manager.get_model_status()
            return model_status
            
        except Exception as e:
            log_error(e, "获取服务状态失败")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_themes(self):
        """获取主题列表"""
        try:
            if self.model_manager is None:
                return {
                    'success': False,
                    'error': '模型管理器未初始化'
                }
            
            themes = self.model_manager.get_theme_keywords()
            return themes
            
        except Exception as e:
            log_error(e, "获取主题列表失败")
            return {
                'success': False,
                'error': str(e)
            }