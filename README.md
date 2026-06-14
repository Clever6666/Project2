# CIFAR-10 CNN / VGG / BatchNorm 实验对比

本项目基于 PyTorch 实现 CIFAR-10 图像分类任务，并对不同网络结构与训练策略进行对比分析。

---

## 项目目标

- 分析网络深度对分类性能的影响  
- 研究 Batch Normalization 的作用  
- 对比不同优化器、激活函数与正则化方法  
- 分析训练稳定性与 loss landscape  

---

## 数据集

- CIFAR-10：10类，32×32彩色图像  
- 训练集：50,000  
- 测试集：10,000  

数据增强：
- Random Crop（padding=4）
- Random Horizontal Flip
- Normalize（0.5, 0.5）

---

## 模型结构

### CNN 系列
- CNN3（浅层）
- CNN4（最佳）
- CNN（标准CNN）
- Residual CNN（残差结构）

### VGG 系列
- VGG-A
- VGG-A + Batch Normalization

BN结构：
Conv → BN → ReLU → Pool

实验结果

### CNN结构对比

| Model | Test Accuracy |
|------|--------------|
| CNN3 | ~0.898 |
| CNN4 | **~0.911** |
| CNN | ~0.906 |
| Residual CNN | ~0.907 |

👉 CNN4 表现最佳

---

### VGG vs BN

| Model | Validation Accuracy |
|------|---------------------|
| VGG-A | 0.776 |
| VGG-A + BN | **0.831** |

👉 BN 提升约 5%

---

### 优化器对比

- SGD：收敛较慢  
- SGD + Momentum：最稳定  
- Adam：收敛快但泛化略弱  
- AdamW：具备正则化优势  

---

### 正则化效果

- Weight Decay = 5e-4 最优  
- Dropout 有效缓解过拟合  
- BatchNorm 提升稳定性与泛化能力  

---

## 📈 主要结论

- BatchNorm 显著提升性能与稳定性  
- CNN4 为最佳基础模型  
- SGD + Momentum 最稳定优化器  
- 模型性能由结构 + 优化共同决定  

---

## 🧪 运行方式

```bash
cd Cifar10_nn_experiment
pip install torch torchvision matplotlib numpy tqdm
python train.py
python experiment_runner.py

cd VGG_BatchNorm
python VGG_Loss_Landscape.py

## 模型权重详见https://github.com/Clever6666/model_weights
