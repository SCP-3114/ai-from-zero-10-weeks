# 第 9 周｜比较实验而不挑最好看的数字

**目标**：设计可比较实验，读懂验证准确率与训练成本的取舍。预计 5–7 小时。

## 动手

1. 从同一配置运行至少两个实验，分别用不同 `--run-name`，只改变一个变量。例如运行 `python -m src.train --config configs/fashion_mnist.json --run-name week09-baseline`，再运行 `python -m src.train --config configs/fashion_mnist.json --learning-rate 0.0005 --run-name week09-lr`。完整实验保持相同 epochs；一轮运行仅用于流程检查。
2. 运行 `python -m week09_experiments.compare`，打开 `results/week09_comparison.csv`。
3. 比较 `best_validation_accuracy`、`training_seconds`、`epochs`、`seed`。不同训练轮数或不同数据划分的结果不能直接归因于学习率。

## 练习与验收

- 写一张实验卡：问题、假设、唯一改动、固定条件、观察值、结论和局限。
- 如果差异很小，再用不同 seed 重复；不要把单次随机波动写成确定性结论。
- 验收：CSV 至少有两行可比结果；结论只使用实际运行的指标；解释为什么最终测试集不能参与挑选配置。

**注意**：汇总脚本只收集 `src.train` 生成的 Fashion-MNIST 结果，不混入第 4 周 CIFAR-10 的指标。
