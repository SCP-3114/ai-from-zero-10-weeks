# 第 4 周｜CNN 与公平对照实验

**目标**：理解卷积为何适合图像，并学会一次只改一个变量。预计 5–8 小时。

## 动手

1. 阅读 `week04_cnn/model.py`，比较 MLP 展平图像与 CNN 保留空间结构的处理方式。
2. 阅读 `week04_cnn/data.py`，找出训练增强与验证/测试预处理的区别。
3. 首先运行一轮环境检查：`python -m week04_cnn.train --experiment baseline --epochs 1`。首次会下载 CIFAR-10，约 170 MB。
4. 时间允许时运行完整 baseline：`python -m week04_cnn.train --experiment baseline`，再运行 `exp_augmentation`。对照 `results/week04/experiments.csv`。
5. 阅读 `week04_cnn/EXPERIMENTS.md` 的实验假设与解释。

## 练习与验收

- 写出 CIFAR-10 与 Fashion-MNIST 在类别、通道数、图片尺寸上的差异。
- 说明 `exp_augmentation` 相比 baseline 改了什么；为什么不能同时改学习率和模型后把效果归给增强。
- 验收：至少运行 baseline 一轮；能用实验表报告验证指标和耗时，不能凭空填写未运行的结果。

**资源提示**：CPU 较慢时保留一轮检查即可；完整实验可分批运行。不要在训练前承诺某个准确率。
