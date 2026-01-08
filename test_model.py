from src.model_manager import get_model_manager

# 测试模型管理器
manager = get_model_manager()

# 显示模型状态
print('模型状态:')
status = manager.get_model_status()
for model, state in status.items():
    print(f'  {model}: {state}')

# 测试预测
test_text = '今天下载了一个新的社交软件，感觉还不错，界面很友好'
print(f'\n测试文本: {test_text}')

result = manager.predict_theme(test_text)
print(f'预测结果: {result}')

if result.get('success'):
    print(f'预测主题: {result.get("theme")}')
    print(f'置信度: {result.get("confidence", 0):.2f}')
    print(f'关键词: {result.get("keywords", [])}')
else:
    print(f'预测失败: {result.get("error")}')

print('\n测试完成！')