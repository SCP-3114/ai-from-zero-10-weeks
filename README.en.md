# AI from Zero: A 10-Week Hands-On Course

[中文 README](README.md) · [Week-by-week lessons](#ten-week-path)

This course is for complete beginners. Start by running a Python program, then learn NumPy, a neural network with manual gradients, PyTorch, CNNs, reproducible experiments, error analysis, robustness, and single-image inference. Finish with a small reproducible project. Plan for about 4–8 hours per week, with extra time for Week 10. The existing code is your learning material; you do not need to write everything from scratch. New to the vocabulary? Start with the [five-minute concept primer](lessons/00-concepts.md) (Chinese).

## Before you begin

- No machine-learning background is needed. Basic arithmetic helps; you can learn the Python syntax along the way.
- Install Python 3.11–3.13 and allow roughly 4 GB of free disk space. A CPU is enough; a GPU is optional. Dataset downloads require internet access.
- Run commands from the project root, the folder containing this README. Replace placeholders such as `results/<run-name>/best_model.pt` with real paths.
- The detailed weekly lessons are in Chinese; this README provides the complete English entry path and commands.

## First run in about 15 minutes

1. Install Python. On Windows, select **Add Python to PATH** during installation. Check with `python --version` (`python3 --version` on Linux/macOS).
2. Open a terminal in this project folder. On Windows, type `powershell` in the folder's File Explorer address bar; elsewhere, use `cd /path/to/summer-ai-foundation-main`.
3. Create a virtual environment, install packages, and run your first module:

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

The rest of this README abbreviates the environment's Python as `python`. On Windows, substitute `.\.venv\Scripts\python.exe`; on Linux/macOS, substitute `.venv/bin/python`. You can also activate the virtual environment first. For a specific CUDA build, use the [official PyTorch installer](https://pytorch.org/get-started/locally/).

**Success check:** you see student information and score statistics without `ModuleNotFoundError`. Then run `python -m pytest -v` to check the project.

## Ten-week path

| Week | Topic | Lesson | Deliverable |
|---|---|---|---|
| 1 | Python, terminal, CSV, matrices | [Week 1](lessons/01-get-started.md) | Run four basic exercises |
| 2 | NumPy, gradients, manual MLP | [Week 2](lessons/02-numpy-and-gradients.md) | Explain backpropagation and a falling loss curve |
| 3 | End-to-end PyTorch training | [Week 3](lessons/03-pytorch.md) | Save and reload a checkpoint |
| 4 | CNNs and fair CIFAR-10 comparisons | [Week 4](lessons/04-cnn.md) | Run a baseline and record the changed variable |
| 5 | Configs, logs, reproducibility, tests | [Week 5](lessons/05-reproducibility.md) | Reconstruct a run from saved artifacts |
| 6 | Confusion matrix and class errors | [Week 6](lessons/06-error-analysis.md) | Identify weak classes and misclassification patterns |
| 7 | Noise and robustness | [Week 7](lessons/07-robustness.md) | Plot accuracy against noise strength |
| 8 | Single-image inference | [Week 8](lessons/08-inference.md) | Report the top three classes for a local image |
| 9 | Comparing runs | [Week 9](lessons/09-experiments.md) | Support a conclusion with comparable runs |
| 10 | Capstone | [Week 10](lessons/10-capstone.md) | Deliver a reproducible report with failure analysis |

Each lesson has goals, steps, exercises, and acceptance criteria. Work at your own pace. If full training is slow, use `--epochs 1` to check the pipeline first, then schedule a full run. Weeks 6–9 require a **Fashion-MNIST** checkpoint from Week 3 or 5; a Week 4 CIFAR-10 checkpoint is incompatible.

## Command reference

```text
python -m exercises.csv_statistics
python -m week02_numpy_nn.linear_regression
python -m week02_numpy_nn.mlp_classifier
python -m src.train --config configs/fashion_mnist.json --epochs 1
python -m src.evaluate --checkpoint results/<run-name>/best_model.pt
python -m week04_cnn.train --experiment baseline --epochs 1
python -m week06_evaluation.report --checkpoint results/<run-name>/best_model.pt
python -m week07_robustness.noise --checkpoint results/<run-name>/best_model.pt
python -m week08_inference.predict --checkpoint results/<run-name>/best_model.pt --image path/to/image.png
python -m week09_experiments.compare
python -m pytest -v
```

Each training run saves its configuration, history, summary, best checkpoint, log, and curves in a separate `results/` subfolder. Week 6 writes a confusion-matrix JSON and per-class CSV; Week 7 writes a noise-results CSV; Week 9 writes a run-comparison CSV. Generated results and downloaded data are ignored by Git. See [Week 4 experiment notes](week04_cnn/EXPERIMENTS.md) for the CIFAR-10 comparison plan.

## How to learn and troubleshoot

For each exercise: **run it, predict a change, change one thing, rerun, and record what happened**. Read the last line of an error first, then check the current directory, Python environment, input files, tensor shapes, and device. If loss becomes `nan`, inspect the learning rate. If training accuracy is much higher than validation accuracy, investigate overfitting. In PowerShell, keep commands on one line instead of using Bash's `\` line continuation.

If time is limited, Weeks 1, 2, 3, 5, 6, and 10 make a shorter complete path. CIFAR-10 in Week 4 requires a larger download and more compute. Report only results you actually ran. Use validation data to choose settings and reserve test data for final evaluation.
