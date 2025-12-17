#!/usr/bin/env python3
"""
应用启动脚本
用于启动主题分类Web应用
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['PYTHONPATH'] = str(project_root)

def main():
    """主函数"""
    try:
        # 导入并启动应用
        from src.app import create_app
        
        # 创建应用实例
        app = create_app('development')
        
        print("=" * 50)
        print("🚀 主题分类工具启动中...")
        print("=" * 50)
        print(f"📁 项目根目录: {project_root}")
        print(f"🌐 访问地址: http://127.0.0.1:5000")
        print("=" * 50)
        
        # 启动Flask应用
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()