#!/usr/bin/env python3
"""
模拟浏览器上传Excel文件到Web服务器
"""
import requests
import os
from pathlib import Path

def test_upload_to_server():
    """测试上传文件到Web服务器"""
    print("=== 测试文件上传到Web服务器 ===")
    
    # 查找测试Excel文件
    data_dir = Path("data/raw/uploads")
    excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    
    if not excel_files:
        print("❌ 没有找到Excel测试文件")
        # 创建一个测试Excel文件
        test_file = data_dir / "test_upload.xlsx"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建简单的Excel文件内容
        import pandas as pd
        df = pd.DataFrame({
            '评论内容': [
                '这个产品质量很好，值得购买',
                '服务态度不错，但是物流有点慢',
                '价格合理，性价比很高',
                '包装很精美，产品也很满意',
                '客服回复很及时，解决问题很专业'
            ]
        })
        df.to_excel(test_file, index=False, engine='openpyxl')
        excel_files = [test_file]
        print(f"✅ 创建测试文件: {test_file}")
    
    # 测试上传每个Excel文件
    upload_url = "http://127.0.0.1:5000/upload"
    
    for excel_file in excel_files:
        print(f"\n--- 测试上传: {excel_file.name} ---")
        
        try:
            with open(excel_file, 'rb') as f:
                files = {'file': (excel_file.name, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                
                response = requests.post(upload_url, files=files, timeout=30)
                
                print(f"状态码: {response.status_code}")
                print(f"响应头: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"✅ 上传成功: {result}")
                    except:
                        print(f"✅ 上传成功，响应内容: {response.text[:200]}...")
                else:
                    print(f"❌ 上传失败: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确保服务器在 http://127.0.0.1:5000 运行")
        except Exception as e:
            print(f"❌ 上传出错: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_upload_to_server()