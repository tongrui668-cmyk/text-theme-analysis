# 模型部署指南

## 概述

本指南详细介绍了主题模型的部署流程，包括自动部署脚本的使用方法、部署步骤、监控策略和故障排查等内容。

## 部署架构

### 目录结构

```
├── data/
│   ├── models/              # 模型文件目录
│   │   ├── deployments/     # 部署历史和报告
│   │   ├── monitoring/      # 性能监控日志
│   │   ├── lda_model.pkl    # 当前部署的LDA模型
│   │   ├── count_vectorizer.pkl  # Count向量化器
│   │   ├── vectorizer.pkl   # TF-IDF向量化器
│   │   ├── theme_classification_model.pkl  # 主题分类器
│   │   ├── theme_keywords.pkl  # 主题关键词
│   │   └── optimized_topics.json  # 优化后的主题名称
│   └── raw/                 # 原始数据目录
├── training_new/
│   ├── scripts/
│   │   └── deploy_model.py  # 自动部署脚本
│   ├── src/                 # 源代码目录
│   └── tests/               # 测试目录
```

## 自动部署流程

### 部署步骤

1. **数据准备**：
   - 加载 `../data/raw/评论和正文.xlsx` 文件
   - 执行文本预处理（清洗、分词、去停用词）
   - 过滤无效记录

2. **模型训练**：
   - 评估最佳主题数（基于困惑度）
   - 训练最终LDA模型
   - 保存模型文件

3. **主题优化**：
   - 自动优化主题名称
   - 生成清晰、简洁的主题标签

4. **模型评估**：
   - 计算困惑度、主题一致性等指标
   - 检测性能退化情况
   - 生成性能报告

5. **模型比较**：
   - 与现有部署的模型比较性能
   - 基于综合得分决定是否部署

6. **部署决策**：
   - 如果新模型性能更优：执行部署
   - 如果现有模型性能更优：跳过部署
   - 如果无现有模型：执行部署

7. **部署执行**：
   - 备份现有模型（如需部署）
   - 复制新模型到部署目录
   - 保存优化后的主题名称

8. **报告生成**：
   - 生成详细的部署报告
   - 保存部署历史
   - 输出部署结果

### 部署决策逻辑

| 情况 | 决策 | 操作 |
|------|------|------|
| 新模型性能 > 现有模型 | 部署新模型 | 备份现有模型，部署新模型 |
| 新模型性能 ≤ 现有模型 | 不部署 | 保持现有模型不变 |
| 无现有模型 | 部署新模型 | 直接部署新模型 |

## 使用指南

### 运行部署脚本

```bash
# 在 training_new 目录下执行
python scripts/deploy_model.py
```

### 脚本参数

部署脚本支持以下参数（通过代码修改）：

- **new_data**：可选，新数据用于模型训练和评估
- **config_updates**：可选，配置更新参数

示例：
```python
# 传入新数据部署
deployment = deployer.deploy_model(new_data=new_text_data)

# 传入配置更新部署
deployment = deployer.deploy_model(config_updates={'n_components': 10})
```

### 查看部署状态

```python
from scripts.deploy_model import ModelDeployer

deployer = ModelDeployer()
status = deployer.get_deployment_status()
print(status)
```

### 回滚部署

```python
from scripts.deploy_model import ModelDeployer

deployer = ModelDeployer()
# 回滚到上一次部署
rollback_result = deployer.rollback_deployment()
print(rollback_result)

# 回滚到指定部署
rollback_result = deployer.rollback_deployment(deployment_id="deploy_20260111_233020")
print(rollback_result)
```

## 性能监控

### 监控指标

- **困惑度**：模型对数据的拟合程度，值越低越好
- **主题一致性**：主题内关键词的相关程度，值越高越好
- **主题清晰度**：主题词分布的清晰度，值越高越好
- **性能退化**：与历史性能相比的变化情况

### 监控日志

性能监控日志会保存到 `../data/models/monitoring/` 目录中，命名格式为 `performance_report_YYYYMMDD_HHMMSS.json`。

### 监控告警

当检测到性能退化时，系统会生成告警信息，包括：
- 性能退化的具体指标
- 建议的解决方案
- 详细的性能报告

## 部署报告

### 报告结构

部署报告包含以下内容：

- **部署信息**：部署ID、时间戳、路径
- **模型信息**：新模型路径、当前模型路径
- **部署决策**：是否部署、决策理由
- **模型性能**：困惑度、一致性、主题分布、退化检测
- **优化主题**：优化后的主题名称和关键词
- **配置更新**：应用的配置更新

### 查看报告

部署报告保存在 `../data/models/deployments/{deployment_id}/deployment_report.json` 文件中。

示例报告：
```json
{
  "deployment_id": "deploy_20260111_233020",
  "timestamp": "2026-01-11T23:30:20.123456",
  "deployment_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\deployments\\deploy_20260111_233020",
  "new_model_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\lda_model.pkl",
  "current_model_path": "D:\\cwu\\job\\experence\\profession\\Mission3\\data\\models\\lda_model.pkl",
  "deployment_decision": {
    "deploy_new": false,
    "reason": "现有模型性能更优"
  },
  "model_performance": {
    "timestamp": "2026-01-11T23:30:20.123456",
    "performance_metrics": {
      "perplexity": 1.0,
      "coherence": 0.6473,
      "topic_distribution": {
        "n_topics": 7,
        "topic_indices": [0, 1, 2, 3, 4, 5, 6]
      },
      "degradation": {
        "detected": false,
        "reasons": []
      }
    },
    "recommendations": [
      "模型性能稳定，无需立即更新",
      "困惑度较高，建议调整模型参数"
    ]
  },
  "optimized_topics": {},
  "config_updates": null
}
```

## 部署历史

部署历史保存在 `../data/models/deployments/deployment_history.json` 文件中，记录了所有部署操作的详细信息。

## 故障排查

### 常见问题

1. **部署失败**：
   - 检查数据文件是否存在
   - 检查模型文件路径权限
   - 查看部署日志中的错误信息

2. **性能评估失败**：
   - 检查模型文件是否损坏
   - 确保数据格式正确
   - 检查内存使用情况

3. **回滚失败**：
   - 检查部署历史是否存在
   - 确保目标部署的模型文件存在
   - 检查文件权限

4. **性能退化**：
   - 检查数据质量
   - 调整模型参数
   - 重新训练模型

### 日志查看

- **部署日志**：控制台输出和部署报告
- **性能日志**：`../data/models/monitoring/` 目录下的JSON文件
- **训练日志**：`logs/train_log_new.txt` 文件

## 最佳实践

### 部署前准备

1. **数据验证**：
   - 确保数据文件存在且格式正确
   - 检查数据质量和完整性
   - 验证数据预处理结果

2. **环境检查**：
   - 确保所有依赖已安装
   - 检查磁盘空间是否充足
   - 验证文件权限设置

3. **配置审查**：
   - 检查配置参数是否合理
   - 验证路径设置是否正确
   - 确认模型保存目录存在

### 部署后操作

1. **性能验证**：
   - 检查部署报告中的性能指标
   - 验证模型预测结果
   - 监控系统运行状态

2. **监控设置**：
   - 定期运行性能监控
   - 设置性能告警阈值
   - 建立监控日志分析流程

3. **备份策略**：
   - 定期备份模型文件
   - 保存部署历史和报告
   - 建立灾难恢复计划

## 自动化建议

### 持续部署

1. **定时部署**：
   - 设置定时任务，定期执行部署脚本
   - 基于新数据自动更新模型

2. **CI/CD集成**：
   - 将部署脚本集成到CI/CD流程中
   - 实现代码变更后的自动部署

3. **监控自动化**：
   - 自动化性能监控和告警
   - 建立自动回滚机制

### 监控自动化

1. **性能监控**：
   - 定期运行模型性能评估
   - 自动生成性能报告

2. **告警机制**：
   - 设置性能退化阈值
   - 实现邮件或消息通知

3. **自动修复**：
   - 当检测到性能退化时，自动触发重新训练
   - 实现智能参数调整

## 版本管理

### 模型版本控制

1. **版本命名**：
   - 部署ID格式：`deploy_YYYYMMDD_HHMMSS`
   - 基于时间戳的版本管理

2. **版本回滚**：
   - 支持回滚到任意历史版本
   - 保留完整的部署历史

3. **版本比较**：
   - 支持不同版本模型的性能比较
   - 提供版本差异分析

## 安全建议

1. **文件权限**：
   - 限制模型文件的访问权限
   - 确保部署脚本的执行权限

2. **数据安全**：
   - 保护原始数据的机密性
   - 避免在日志中记录敏感信息

3. **部署安全**：
   - 验证部署脚本的完整性
   - 确保部署过程的可审计性

## 总结

本部署指南提供了主题模型部署的完整流程和最佳实践。通过使用自动部署脚本，您可以实现模型的快速更新和性能监控，确保模型始终保持最佳状态。

如需进一步的帮助或有任何问题，请参考项目文档或联系开发团队。