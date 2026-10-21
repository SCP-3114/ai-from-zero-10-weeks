# 第 5 周｜把训练做成可复现的工程

**目标**：理解配置、随机种子、日志、checkpoint、测试与结果目录。预计 4–6 小时。

## 动手

1. 阅读 `configs/fashion_mnist.json`，记录 seed、epochs、batch size、learning rate、hidden size、dropout 的含义。
2. 阅读 `src/train.py`，追踪配置如何写进每个结果目录；找出何时保存最佳 checkpoint。
3. 运行 `python -m pytest -v`。测试应离线完成，不需下载数据。
4. 运行 `python -m src.train --config configs/fashion_mnist.json --epochs 1 --run-name week05-a`，再运行同样命令但把名称改为 `week05-b`；不要覆盖已有目录。比较两个 `config.json` 和指标。

## 练习与验收

- 用 `python -m src.train --config configs/fashion_mnist.json --epochs 1 --learning-rate 0.0005 --run-name week05-lr` 做第三次运行，记录变量、验证准确率、训练秒数。
- 解释同种子为什么有助于复现，以及硬件/算子仍可能带来小差异。
- 验收：能从一个结果目录重建实验条件；能解释训练、验证、测试指标的区别；测试通过。

**注意**：结果目录由程序生成，`results/` 已被 Git 忽略；交作业时应附配置、指标和必要图片，而非大型原始数据。
