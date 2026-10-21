# 第 8 周｜把模型用于一张新图片

**目标**：理解训练与推理的区别，能加载 checkpoint、复用预处理、输出前三类别。预计 4–6 小时。

## 动手

1. 准备一张衣物图片（PNG/JPEG）；建议先用灰度、居中、深色背景、浅色衣物。模型训练图片是 28×28 灰度 Fashion-MNIST，真实照片可能效果较差。
2. 运行 `python -m week08_inference.predict --checkpoint results/<实验名>/best_model.pt --image path/to/your_image.png`。
3. 如果图片是浅色背景、深色衣物，加入 `--invert` 再比较。
4. 阅读 `week08_inference/predict.py`，找出转灰度、缩放填充、转 Tensor、归一化、`model.eval()`、`inference_mode()` 与 softmax。

## 练习与验收

- 说明为什么推理必须复用训练时的归一化；为什么输出概率高不一定意味着模型可靠。
- 记录一张成功和一张失败图片，说明背景、裁剪、拍摄角度可能造成的差异。
- 验收：命令能输出前三类及概率；能解释输入张量形状 `(1, 1, 28, 28)`。

**隐私提示**：脚本只在本机读取图片，不上传图片。
