"""
重构后的Flask应用主文件
使用Blueprint分离路由，使用服务类分离业务逻辑
"""
import os
import sys
from pathlib import Path
from flask import Flask
from werkzeug.exceptions import NotFound

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.logger import setup_logger, log_info
from src.routes import main_bp
from src.config import config
from src.model_manager import get_model_manager

# 设置日志
setup_logger()

def create_app(config_name='development'):
    """应用工厂函数"""
    # 设置模板目录
    template_folder = Path(__file__).parent.parent / 'static' / 'templates'
    static_folder = Path(__file__).parent.parent / 'static'
    
    app = Flask(__name__, 
                template_folder=str(template_folder),
                static_folder=str(static_folder))
    
    # 加载配置
    app.config.from_object(config[config_name])
    
    # 确保上传目录存在
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # 注册蓝图
    app.register_blueprint(main_bp)
    
    # 注册错误处理器
    register_error_handlers(app)
    
    # 初始化模型
    init_models()
    
    log_info("应用初始化完成")
    return app

def register_error_handlers(app):
    """注册错误处理器"""
    @app.errorhandler(404)
    def not_found_error(error):
        from flask import render_template
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        from flask import render_template
        return render_template('500.html'), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        log_info(f"未处理的异常: {str(e)}")
        from flask import render_template, request
        return render_template('error.html', error=str(e)), 500

def init_models():
    """初始化模型"""
    try:
        model_manager = get_model_manager()
        log_info("模型初始化完成")
    except Exception as e:
        log_info(f"模型初始化失败: {str(e)}")



# 创建应用实例
def create_app_instance():
    """创建应用实例（兼容性函数）"""
    return create_app()

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)