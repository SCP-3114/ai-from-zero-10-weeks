# 从零开始学 AI：10 周动手教程

[English README](README.en.md) · [10 周课程目录](#十周路线图)

这是一条面向**完全初学者**的实践路线：从运行 Python 程序开始，逐步学习 NumPy、手写神经网络、PyTorch、CNN、实验复现、错误分析、鲁棒性和单图推理，最后完成一个可复现的小项目。每周约 4–8 小时；第 10 周可多留一些时间。已有代码是学习材料，不需要从空白文件写起。第一次接触这些词，先读[五分钟概念入门](lessons/00-concepts.md)。

## 你需要准备什么

- 能安装软件和使用终端即可；不要求高等数学或机器学习经验。先会四则运算和简单函数图像会更轻松。
- Python 3.11–3.13、约 4 GB 可用磁盘空间；CPU 可运行，GPU 可选。下载数据集时需要网络。安装 PyTorch 可能耗时。
- 在本目录执行所有命令。命令里的 `results/<实验名>/best_model.pt` 是占位符，必须换成实际路径。
- 跟着 [第 1 周](lessons/01-get-started.md) 学，不必先看懂所有文件。

## 从零开始：15 分钟跑通第一条命令

1. 安装 Python。Windows 安装时勾选 **Add Python to PATH**。打开终端，运行 `python --version`；Linux/macOS 可用 `python3 --version`。
2. 进入本项目目录，即含有这个 README 的目录。Windows PowerShell 可在资源管理器的目录地址栏输入 `powershell`；Linux/macOS 使用 `cd /你的/路径/summer-ai-foundation-main`。
3. 创建虚拟环境并安装依赖。以下命令任选与你的系统对应的一组：

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.python_basics
```

**Linux/macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.python_basics
```

下文用 `python` 简写：Windows 可替换为 `.\.venv\Scripts\python.exe`，Linux/macOS 可替换为 `.venv/bin/python`。也可以先激活虚拟环境再直接使用 `python`。如果需要特定 CUDA 版本，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择安装命令。

**成功标志**：程序打印学生信息和成绩统计，且没有 `ModuleNotFoundError`。接着运行 `python -m pytest -v`，确认代码测试通过。

## 十周路线图

| 周 | 学习主题 | 从哪开始 | 完成标志 |
|---|---|---|---|
| 1 | Python、终端、CSV、矩阵 | [第 1 周教程](lessons/01-get-started.md) | 能运行四个基础练习 |
| 2 | NumPy、梯度、手写 MLP | [第 2 周教程](lessons/02-numpy-and-gradients.md) | loss 曲线下降，能解释反向传播 |
| 3 | PyTorch 训练闭环 | [第 3 周教程](lessons/03-pytorch.md) | 保存 checkpoint 并重新评估 |
| 4 | CNN、CIFAR-10、公平对照 | [第 4 周教程](lessons/04-cnn.md) | 运行 baseline 并记录实验变量 |
| 5 | 配置、日志、复现、测试 | [第 5 周教程](lessons/05-reproducibility.md) | 能从结果目录还原实验条件 |
| 6 | 混淆矩阵与类别错误 | [第 6 周教程](lessons/06-error-analysis.md) | 找出薄弱类别及误判方向 |
| 7 | 噪声与鲁棒性 | [第 7 周教程](lessons/07-robustness.md) | 画出噪声强度与准确率关系 |
| 8 | 单图推理 | [第 8 周教程](lessons/08-inference.md) | 对本地图片输出前三类别 |
| 9 | 实验比较 | [第 9 周教程](lessons/09-experiments.md) | 用同条件实验表支持结论 |
| 10 | 结业项目 | [第 10 周教程](lessons/10-capstone.md) | 交付可复现报告和失败分析 |

每篇教程包含目标、动手步骤、练习、验收和常见问题。可以按自己的节奏学；如果第 3/4 周训练耗时，先做 `--epochs 1` 环境检查，再安排完整训练。第 6–9 周依赖第 3 或 5 周生成的 **Fashion-MNIST** checkpoint，不能拿第 4 周的 CIFAR-10 checkpoint 运行。

## 核心命令速查

```text
python -m exercises.csv_statistics
python -m week02_numpy_nn.linear_regression
python -m week02_numpy_nn.mlp_classifier
python -m src.train --config configs/fashion_mnist.json --epochs 1
python -m src.evaluate --checkpoint results/<实验名>/best_model.pt
python -m week04_cnn.train --experiment baseline --epochs 1
python -m week06_evaluation.report --checkpoint results/<实验名>/best_model.pt
python -m week07_robustness.noise --checkpoint results/<实验名>/best_model.pt
python -m week08_inference.predict --checkpoint results/<实验名>/best_model.pt --image path/to/image.png
python -m week09_experiments.compare
python -m pytest -v
```

训练会将 `config.json`、`history.json`、`summary.json`、`best_model.pt`、日志和曲线保存在独立的 `results/` 子目录。第 6 周会生成混淆矩阵 JSON 与逐类别 CSV；第 7 周生成噪声测试 CSV；第 9 周生成比较 CSV。这些运行产物和下载数据默认被 Git 忽略。第 4 周实验设计和解释见 [EXPERIMENTS.md](week04_cnn/EXPERIMENTS.md)。

## 学习方法与求助线索

每周按“**先运行 → 预测改动结果 → 只改一处 → 再运行 → 记录观察**”学习。报错时先读最后一行，再查当前目录、Python 环境、输入文件、Tensor 形状和设备。训练 loss 为 `nan` 时先检查学习率；训练准确率远高于验证准确率时考虑过拟合。Windows PowerShell 不使用 Bash 的 `\` 换行符；最简单是把命令写在一行。

若时间有限，第 1、2、3、5、6、10 周构成最小完整路径；第 4 周 CIFAR-10 下载和训练更耗资源。所有报告只填自己实际运行的结果，不把示例数字当作目标。测试集用于最终评估，不用于挑选超参数。
