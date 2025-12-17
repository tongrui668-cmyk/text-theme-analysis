"""
修复后的日志系统配置
解决多进程日志文件占用问题
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from src.config import config

class Logger:
    """统一日志管理类 - 修复多进程问题"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """获取或创建日志器"""
        if name is None:
            name = __name__
            
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """创建日志器 - 修复多进程冲突"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config['default'].LOG_CONFIG['log_level']))
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建日志目录
        log_file = config['default'].LOG_CONFIG['log_file']
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 为每个进程创建独立的日志文件
        process_id = os.getpid()
        process_log_file = log_file.parent / f"app_{process_id}.log"
        
        # 文件处理器（简化版，不使用轮转避免冲突）
        file_handler = logging.FileHandler(
            process_log_file,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            f'%(asctime)s - PID:{process_id} - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def setup_flask_logging(cls, app):
        """为Flask应用配置日志"""
        # 禁用Flask默认日志
        app.logger.handlers.clear()
        
        # 使用自定义日志器
        flask_logger = cls.get_logger('flask')
        app.logger = flask_logger
        
        # 配置其他Flask相关日志
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

def log_function_call(func):
    """装饰器：记录函数调用"""
    def wrapper(*args, **kwargs):
        logger = Logger.get_logger()
        logger.debug(f"调用函数: {func.__name__}, 参数: args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
    
    return wrapper

def log_error(error: Exception, context: str = ""):
    """记录错误信息"""
    logger = Logger.get_logger()
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=True)

def log_info(message: str):
    """记录信息"""
    logger = Logger.get_logger()
    logger.info(message)

def log_warning(message: str):
    """记录警告"""
    logger = Logger.get_logger()
    logger.warning(message)

def log_debug(message: str):
    """记录调试信息"""
    logger = Logger.get_logger()
    logger.debug(message)

# 清理函数
def cleanup_logs():
    """清理日志文件"""
    log_dir = Path("logs")
    if log_dir.exists():
        # 删除旧的进程特定日志文件
        for log_file in log_dir.glob("app_*.log"):
            try:
                if log_file.stat().st_size == 0:  # 删除空日志文件
                    log_file.unlink()
            except:
                pass

# 初始化日志系统
def setup_logger():
    """设置日志系统（兼容性函数）"""
    init_logging()

def init_logging():
    """初始化日志系统"""
    # 清理旧日志
    cleanup_logs()
    
    # 创建日志目录
    log_dir = config['default'].LOG_CONFIG['log_file'].parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试日志器
    test_logger = Logger.get_logger('test')
    test_logger.info("日志系统初始化完成")

if __name__ == '__main__':
    init_logging()
    
    # 测试日志功能
    logger = Logger.get_logger('main')
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    logger.debug("这是一条调试信息")