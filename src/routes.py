"""
路由模块
将所有路由定义集中管理
"""
from flask import Blueprint, request, render_template, jsonify, redirect, url_for, flash
from pathlib import Path
import pandas as pd
from werkzeug.utils import secure_filename

from src.logger import log_info, log_error
from src.services import FileProcessingService, TextAnalysisService

# 创建蓝图
main_bp = Blueprint('main', __name__)

# 创建服务实例
file_service = FileProcessingService()
text_service = TextAnalysisService()

@main_bp.route('/')
def index():
    """主页"""
    try:
        return render_template('index.html')
    except Exception as e:
        log_error(e, "主页加载失败")
        return render_template('error.html', error=str(e)), 500

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """文件上传处理"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            flash('❌ 没有选择文件')
            return redirect(url_for('main.index'))
        
        file = request.files['file']
        
        # 检查文件名
        if file.filename == '':
            flash('❌ 没有选择文件')
            return redirect(url_for('main.index'))
        
        # 验证文件格式
        is_allowed, format_message = file_service.allowed_file(file.filename)
        if not is_allowed:
            flash(f'❌ {format_message}')
            return redirect(url_for('main.index'))
        
        # 验证文件大小
        size_valid, size_message = file_service.validate_file_size(file)
        if not size_valid:
            flash(f'❌ {size_message}')
            return redirect(url_for('main.index'))
        
        # 保存文件
        original_filename = file.filename
        filename = secure_filename(original_filename)
        filepath = file_service.save_uploaded_file(file, filename)
        
        log_info(f"文件上传成功: {filename} ({size_message})")
        flash(f'✅ 文件上传成功：{original_filename}')
        
        # 处理文件 - 使用与allowed_file相同的方法获取扩展名
        extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        if extension in {'xlsx', 'xls'}:
            return file_service.process_excel_file(filepath)
        elif extension == 'txt':
            return file_service.process_text_file(filepath)
        elif extension == 'csv':
            return file_service.process_csv_file(filepath)
        else:
            flash('❌ 不支持的文件格式')
            return redirect(url_for('main.index'))
            
    except Exception as e:
        log_error(e, "文件上传处理失败")
        flash(f'❌ 文件处理失败: {str(e)}')
        return redirect(url_for('main.index'))

@main_bp.route('/analyze_text', methods=['POST'])
def analyze_text():
    """直接分析文本"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': '文本内容不能为空'
            })
        
        # 分析文本
        result = text_service.analyze_text(text)
        
        return jsonify(result)
        
    except Exception as e:
        log_error(e, "文本分析失败")
        return jsonify({
            'success': False,
            'error': f'分析失败: {str(e)}'
        })

@main_bp.route('/api/status')
def api_status():
    """API状态接口"""
    try:
        status = text_service.get_service_status()
        
        return jsonify({
            'success': True,
            'app_status': 'running',
            **status
        })
        
    except Exception as e:
        log_error(e, "获取状态失败")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@main_bp.route('/api/themes')
def api_themes():
    """获取主题列表"""
    try:
        themes = text_service.get_themes()
        
        return jsonify({
            'success': True,
            'themes': themes
        })
        
    except Exception as e:
        log_error(e, "获取主题列表失败")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 错误处理
@main_bp.app_errorhandler(404)
def not_found(error):
    """404错误处理"""
    return render_template('error.html', error='页面未找到'), 404

@main_bp.app_errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return render_template('error.html', error='服务器内部错误'), 500

@main_bp.app_errorhandler(Exception)
def handle_exception(e):
    """全局异常处理"""
    log_error(e, "未处理的异常")
    
    if request.is_json:
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500
    else:
        return render_template('error.html', error='服务器内部错误'), 500