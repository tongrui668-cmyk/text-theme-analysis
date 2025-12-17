"""
测试修复后的日志系统
"""
import sys
import time
import multiprocessing
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, 'src')

def test_logging_in_process(process_id):
    """在单个进程中测试日志"""
    from src.logger import Logger, init_logging
    
    # 初始化日志系统
    init_logging()
    
    logger = Logger.get_logger(f'test_process_{process_id}')
    
    # 写入多条日志
    for i in range(5):
        logger.info(f"进程 {process_id} - 日志消息 {i+1}")
        time.sleep(0.1)
    
    logger.warning(f"进程 {process_id} - 这是一条警告")
    logger.error(f"进程 {process_id} - 这是一条错误")
    
    return f"进程 {process_id} 完成"

def main():
    """主测试函数"""
    print("=== 测试修复后的日志系统 ===")
    
    # 清理现有日志
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("app_*.log"):
            try:
                log_file.unlink()
                print(f"删除旧日志文件: {log_file}")
            except:
                pass
    
    # 创建多个进程同时写入日志
    num_processes = 3
    processes = []
    
    print(f"\n启动 {num_processes} 个进程同时测试日志...")
    
    for i in range(num_processes):
        p = multiprocessing.Process(target=test_logging_in_process, args=(i+1,))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("\n所有进程完成，检查生成的日志文件...")
    
    # 检查生成的日志文件
    log_files = list(log_dir.glob("app_*.log"))
    print(f"生成了 {len(log_files)} 个日志文件:")
    
    for log_file in log_files:
        size = log_file.stat().st_size
        print(f"  - {log_file.name}: {size} 字节")
        
        # 读取并显示前几行
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:3]
                for line in lines:
                    print(f"    {line.strip()}")
        except Exception as e:
            print(f"    读取文件出错: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()