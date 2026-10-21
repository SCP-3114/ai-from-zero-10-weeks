# 第 6 周｜从准确率走向错误分析

**目标**：读懂混淆矩阵和逐类别准确率，知道总体准确率掩盖了什么。预计 4–6 小时；需要第 3 或 5 周训练出的 Fashion-MNIST checkpoint。

## 动手

1. 运行 `python -m week06_evaluation.report --checkpoint results/<实验名>/best_model.pt`。
2. 打开同一实验目录下的 `week06_evaluation/confusion_matrix.json` 与 `per_class.csv`。矩阵的**行是真实类别，列是预测类别**。
3. 找准确率最低的两个类别，再找它们最常被误判为哪一类。对照第 3 周生成的 `incorrect_predictions.png`，尝试解释原因。

## 练习与验收

- 手算某类别 `correct / support`，与 CSV 比较。
- 记录一个具体改进假设，例如“衬衫与外套易混淆，可能需要更丰富的衣物轮廓样本”。这是猜想，须用后续实验验证。
- 验收：能说清混淆矩阵的两个轴；报告总体准确率、最弱类别及一个证据支持的假设。

**提醒**：脚本会再次读取官方测试集；本周做错误分析可以，但后续选超参数应以验证集为准，不要反复根据测试集调参。
