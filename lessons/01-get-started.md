# 第 1 周｜把环境搭起来，学会运行 Python

**目标**：能解释“项目根目录、终端、虚拟环境、模块、依赖”是什么，并独立运行四个入门练习。预计 4–6 小时；无需 AI 基础。

## 先理解

Python 文件是程序；终端负责发出运行命令；虚拟环境把本项目的包与其他项目隔开。`python -m 包.模块` 表示从项目根目录按模块运行，因此先用 `pwd`（PowerShell 可用 `Get-Location`）确认当前目录包含 `README.md`。

## 动手

1. 按 [中文 README](../README.md) 的“从零开始”安装 Python、创建虚拟环境并安装依赖。运行 `python --version` 和 `python -m pytest --version` 检查。
2. 运行 `python -m src.python_basics`，阅读 `src/python_basics.py`，改一位学生的年龄再运行，观察输出变化。
3. 运行 `python -m exercises.matrix_operations`，用纸笔算一个 2×2 矩阵乘法，再核对程序输出。
4. 运行 `python -m exercises.cli_logging --epochs 3 --learning-rate 0.001`。再输入 `--help`，理解命令行参数的作用。
5. 运行 `python -m exercises.csv_statistics`。它默认读取 `sample_data/scores.csv`；给示例文件加一行成绩，观察均值变化。

## 练习与验收

- 用自己的话解释列表、字典、函数、`for` 循环各自解决什么问题。
- 修改 `src/python_basics.py` 中的成绩，预测均值后再运行核对。
- 故意把 CSV 中一个数字改成文字，观察报错指向哪一行，随后还原。
- 验收：以上四条命令都能从项目根目录运行；能说明虚拟环境的用途；知道下载到 `data/` 的数据集不应提交到代码仓库。

**常见卡点**：PowerShell 禁止激活脚本时，可直接用 `.\.venv\Scripts\python.exe -m ...` 运行，不必先激活；Linux 使用 `.venv/bin/python`。路径中有空格时加引号。

**下一周**：学习数组和梯度；先看 `week02_numpy_nn/linear_regression.py` 中输入、预测、误差的形状。
