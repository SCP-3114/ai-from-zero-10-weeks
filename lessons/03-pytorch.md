# 第 3 周｜第一次完整训练 PyTorch 模型

**目标**：会使用 Tensor、Dataset、DataLoader、`nn.Module`、自动求导、训练/验证/测试划分。预计 5–8 小时。

## 动手

1. 阅读 `week03_pytorch/data.py`，追踪一张图片如何变成 `(1, 28, 28)` 的 Tensor；批量后形状是 `(B, 1, 28, 28)`。
2. 阅读 `week03_pytorch/model.py` 和 `train.py`，找出 `zero_grad → forward → loss → backward → step`。
3. 从项目根目录运行工程化入口的一轮快速检查：`python -m src.train --config configs/fashion_mnist.json --epochs 1`。首次运行会下载 Fashion-MNIST，CPU 可以完成但需要耐心。
4. 查看新建的 `results/<实验名>/`，读取 `config.json`、`history.json`、`summary.json`、曲线和日志。
5. 将实际路径代入 `python -m src.evaluate --checkpoint results/<实验名>/best_model.pt`，得到测试指标和错误图片。

## 练习与验收

- 解释 `model.train()` 与 `model.eval()`、`torch.inference_mode()` 的作用。
- 解释验证集用来选择模型、测试集只用于最终评估的原因。
- 验收：能指出 checkpoint 在哪里；能重新加载它并输出测试准确率；知道一轮结果只是环境检查，不能代表最终能力。

**常见卡点**：必须在根目录运行 `python -m src.train`；出现下载错误时先检查网络，再重试；没有 GPU 属于正常情况。
