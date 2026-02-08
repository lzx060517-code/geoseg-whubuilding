# 遥感建筑物全智能化三维测图 - 阶段一

本项目是大创项目“遥感建筑物全智能化三维测图与人工智能测图精度评定理论方法研究”的第一阶段产出：**基于边缘感知 CNN 与视觉 Transformer 的建筑物轮廓高精度智能提取**。

## 🚀 项目概况

本仓库代码基于开源遥感分割框架 **GeoSeg** 进行二次开发与适配。我们目前已针对 **WHU Building Dataset** 完成了 Baseline 模型的训练与复现。

### 📊 Baseline 结果 (WHU Dataset)

| 指标 | 结果 |
| --- | --- |
| **Building IoU** | **88.05%** |
| **Building F1** | **93.65%** |
| **Mean IoU** | **93.25%** |
| **Overall Accuracy** | **98.60%** |

---

## 🛠️ 环境配置与安装

代码运行环境建议为 `Python 3.8+`，建议使用 Conda 建立环境。

```bash
# 创建并激活环境
conda create -n geoseg312 python=3.12
conda activate geoseg312

# 安装核心依赖
pip install torch torchvision torchaudio
pip install pytorch-lightning timm opencv-python

```

---

## 📂 快速开始 

### 1. 数据与权重准备

由于 `.gitignore` 策略，大型数据和权重需手动同步：

* **数据集**：请将 WHU 数据集放在 `data/whubuilding/1/` 目录下。
* **权重文件**：请将 `unetformer-r18-whu-1024-e40.ckpt`放入 `model_weights/whu_building/`。

### 2. 运行测试

使用已修复文件名覆盖 Bug 的 `whu_test.py` 进行推理：

```bash
python whu_test.py -c config/whu_config.py -o outputs/test_results --rgb --tta d4 --scales 0.75,1.0,1.25,1.5 --eval

```

---

## 🎯 后续研究路线

目前的 Baseline 仍存在边缘模糊和圆角化问题，后续我们将按照研究计划推进以下改进：

1. **引入边缘感知分支**：在 Encoder 侧引出辅助流，显式监督边界特征。
2. **嵌入 SimAM 注意力**：在特征提取阶段增强关键空间与通道特征的筛选能力。
3. **几何约束 Loss**：在训练后期引入 **Hausdorff Loss**，从几何层面优化轮廓精度。

---

## 📖 鸣谢

* 核心框架：[GeoSeg](https://github.com/WangLibo1995/GeoSeg)
* 基础模型：UNetFormer

---

### 💡 小贴士

* **修改路径**：拿到代码后，第一时间去 `config/whu_config.py` 里把 `data_root` 改成电脑上的绝对路径。
* **查看曲线**：训练日志在 `lightning_logs/` 中，可以使用 `tensorboard --logdir=lightning_logs` 查看收敛过程。

