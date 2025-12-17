import requests
import os
from pathlib import Path

# 文件路径
file_path = Path("data/raw/uploads/-1.xlsx")

if not file_path.exists():
    print(f"文件不存在: {file_path}")
    exit(1)

# 上传文件
url = "http://127.0.0.1:5000/upload"

try:
    with open(file_path, 'rb') as f:
        files = {'file': ('-1.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        
        print(f"正在上传文件: {file_path}")
        print(f"文件大小: {file_path.stat().st_size / 1024:.1f} KB")
        
        response = requests.post(url, files=files)
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("文件上传成功！")
            
            # 检查响应内容是否包含分析结果
            if "成功分析数" in response.text:
                print("响应包含分析结果")
                
                # 尝试提取关键信息
                if "总文本数" in response.text:
                    import re
                    total_match = re.search(r'总文本数：(\d+)', response.text)
                    success_match = re.search(r'成功分析数：(\d+)', response.text)
                    
                    if total_match and success_match:
                        total = int(total_match.group(1))
                        success = int(success_match.group(1))
                        print(f"总文本数: {total}")
                        print(f"成功分析数: {success}")
                        print(f"成功率: {success/total*100:.1f}%" if total > 0 else "成功率: 0%")
            else:
                print("响应不包含分析结果，可能是错误页面")
                print("响应内容前500字符:")
                print(response.text[:500])
        else:
            print(f"上传失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            
except Exception as e:
    print(f"上传过程中出错: {e}")