<!-- 头部徽章 -->
<p align="center">
  <img src="https://img.shields.io/github/stars/Minecraft-JoJo/cat-dog-classifier?style=social" alt="stars">
  <img src="https://img.shields.io/github/forks/Minecraft-JoJo/cat-dog-classifier?style=social" alt="forks">
  <img src="https://img.shields.io/github/license/Minecraft-JoJo/cat-dog-classifier" alt="license">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="pytorch">
</p>

# 🐱🐶 Cat-Dog Classifier

> 一个基于 GoogLeNet (Inception v1) 的猫狗图像分类器，用 PyTorch 实现，支持训练、验证和单张预测。

[![demo](https://img.shields.io/badge/demo-not%20yet-red)](#) [![colab](https://img.shields.io/badge/run%20in-Colab-yellow)](https://colab.research.google.com)

---

## 📖 项目简介

这个项目实现了一个端到端的猫狗分类系统，包括：

- ✅ 自定义数据加载与增强
- ✅ GoogLeNet (Inception v1) 网络架构
- ✅ 完整的训练/验证/测试流程
- ✅ 模型保存与加载
- ✅ 单张图片预测
- ✅ 支持 GPU / CPU 自动切换

---

## 🗂️ 数据集

本项目的训练和验证数据来自 [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)，按照 8:2 划分。

- 训练集：约 7000 张图片（每类 3500）
- 验证集：约 2000 张图片（每类 1000）
- 测试集：约 2000 张图片（每类 1000）

**文件夹结构**：
data/
├── train/
│ ├── cats/ # 猫图片
│ └── dogs/ # 狗图片
├── val/
│ ├── cats/
│ └── dogs/
└── test/
├── cats/
└── dogs/

---

## 🚀 快速开始


### 第一步：克隆项目并进入目录

```bash
git clone https://github.com/Minecraft-JoJo/cat-dog-classifier.git
cd cat-dog-classifier
```
### 第二步：创建并进入环境
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 第三步：安装依赖环境
```bash
pip install -r requirements.txt
```
### 环境要求

- Python 3.10+
- PyTorch 2.0+
- torchvision
- tqdm
- matplotlib
- pandas
- numpy

训练模型
```bash
python train.py
```
预测单张图片
```bash
python single.py
```
或使用 model_test.py 批量预测：

```bash
python model_test.py
```
🏗️ 模型架构
本项目使用 GoogLeNet (Inception v1)，核心组件是 Inception 模块，它通过 1x1、3x3、5x5 卷积和池化并行提取多尺度特征，并在通道维度拼接。

参数量约 500 万

支持 224x224 彩色图像

输出层为全连接层（Softmax）

如果你需要更轻量或更深的模型，可以修改 model.py 中的 class GoogLeNet。

📊 实验结果
模型	验证集准确率	测试集准确率	训练轮数
GoogLeNet (从头训练)	~90%	~90%	50
注意：准确率会因数据集划分和随机种子略有浮动。

📁 项目文件
文件	作用
train.py	训练主脚本
test.py	测试主脚本
model.py	GoogLeNet 和 Inception 定义
config.py	配置参数（路径、学习率等）
single.py	单张图片预测
model_test.py	文件夹批量预测
data/	数据集（需自行下载）
requirements.txt	依赖清单
.gitignore	Git 忽略规则
🖼️ 预测示例
python
# 单张预测
```bash
python single.py
```
# 输出示例：
# 预测的结果: dog，置信度: 0.9812
📄 许可证
本项目采用 MIT License，可自由使用和修改。

🙏 致谢
Kaggle Dogs vs Cats 提供数据集

PyTorch 深度学习框架


⭐ 给个 Star 吧！
如果你觉得这个项目有用，欢迎点个 Star ⭐ 支持一下～
也欢迎 fork 和提 issue。

</final>
