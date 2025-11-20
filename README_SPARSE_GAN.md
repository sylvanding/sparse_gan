# Sparse GAN - 3D体素生成对抗网络

基于 **MinkowskiEngine** 实现的稀疏卷积 GAN，用于生成 3D 医学影像体素数据。

## 🌟 特性

- ✨ **稀疏卷积架构**：使用 MinkowskiEngine 高效处理稀疏 3D 数据
- 🎯 **WGAN-GP 训练**：稳定的 GAN 训练策略，支持梯度惩罚
- 🔄 **灵活的数据加载**：集成 NIfTI 数据集，自动处理密集-稀疏转换
- 📊 **多种输出格式**：支持 NIfTI、NumPy 格式输出
- 🎨 **可视化工具**：内置切片和 3D 可视化功能
- 🔀 **潜在空间插值**：支持线性和球面线性插值

## 📁 文件结构

```
sparse_gan/
├── sparse_gan_config.yaml          # 配置文件
├── sparse_gan_models.py            # 生成器和判别器模型
├── sparse_gan_dataset.py           # 数据加载和预处理
├── sparse_gan_trainer.py           # 训练器
├── sparse_gan_sampling.py          # 采样和可视化
├── train_sparse_gan.py             # 训练启动脚本
├── sample_sparse_gan.py            # 采样启动脚本
├── voxel_nifti_dataset.py          # NIfTI数据集加载器
└── README_SPARSE_GAN.md            # 本文档
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch torchvision

# 安装 MinkowskiEngine
pip install MinkowskiEngine

# 安装其他依赖
pip install monai nibabel pyyaml tensorboard tqdm matplotlib
```

### 2. 准备数据

将 NIfTI 格式的体素数据放入训练和验证目录：

```
/data/nifti/
├── train/
│   ├── sample_001.nii.gz
│   ├── sample_002.nii.gz
│   └── ...
└── val/
    ├── sample_101.nii.gz
    └── ...
```

### 3. 修改配置

编辑 `sparse_gan_config.yaml`：

```yaml
data:
  train_data_dir: "/path/to/your/train"  # 修改为你的训练数据路径
  val_data_dir: "/path/to/your/val"      # 修改为你的验证数据路径
  voxel_size: 64                          # 体素分辨率
```

### 4. 开始训练

```bash
# 基本训练
python train_sparse_gan.py --config sparse_gan_config.yaml

# 指定参数训练
python train_sparse_gan.py \
    --config sparse_gan_config.yaml \
    --train_data_dir /path/to/train \
    --batch_size 4 \
    --num_epochs 200

# 恢复训练
python train_sparse_gan.py \
    --config sparse_gan_config.yaml \
    --resume checkpoints/sparse_gan/checkpoint_latest.pth
```

### 5. 生成样本

```bash
# 随机生成10个样本
python sample_sparse_gan.py \
    --checkpoint checkpoints/sparse_gan/checkpoint_best.pth \
    --num_samples 10 \
    --output_dir outputs/samples

# 生成插值序列
python sample_sparse_gan.py \
    --checkpoint checkpoints/sparse_gan/checkpoint_best.pth \
    --interpolate \
    --num_steps 20 \
    --output_dir outputs/interpolation

# 生成并可视化
python sample_sparse_gan.py \
    --checkpoint checkpoints/sparse_gan/checkpoint_best.pth \
    --num_samples 5 \
    --visualize \
    --format both
```

## 🏗️ 模型架构

### 生成器 (SparseGenerator)

```
潜在向量 (256D)
    ↓
全连接层 → 初始特征网格 (4×4×4, 256通道)
    ↓
稀疏转置卷积 ↑2× → 8×8×8, 128通道
    ↓
稀疏转置卷积 ↑2× → 16×16×16, 64通道
    ↓
稀疏转置卷积 ↑2× → 32×32×32, 32通道
    ↓
稀疏转置卷积 ↑2× → 64×64×64, 16通道
    ↓
输出卷积 → 64×64×64, 1通道 (稀疏)
```

### 判别器 (SparseDiscriminator)

```
输入体素 (64×64×64, 1通道, 稀疏)
    ↓
稀疏卷积 ↓2× → 32×32×32, 16通道
    ↓
稀疏卷积 ↓2× → 16×16×16, 32通道
    ↓
稀疏卷积 ↓2× → 8×8×8, 64通道
    ↓
稀疏卷积 ↓2× → 4×4×4, 128通道
    ↓
稀疏卷积 ↓2× → 2×2×2, 256通道
    ↓
全局池化 + 全连接 → 真/假分数
```

## ⚙️ 配置说明

### GAN 类型

- `wgan-gp`：Wasserstein GAN with Gradient Penalty（推荐，最稳定）
- `vanilla`：标准 GAN（BCE 损失）
- `lsgan`：Least Squares GAN

### 关键参数

```yaml
generator:
  latent_dim: 256              # 潜在向量维度
  channels: [256, 128, 64, 32, 16]  # 每层通道数
  initial_tensor_stride: 32    # 初始稀疏张量步长

discriminator:
  channels: [16, 32, 64, 128, 256]  # 每层通道数

training:
  gan_type: "wgan-gp"          # GAN类型
  n_critic: 5                   # 判别器训练频率
  gradient_penalty_weight: 10.0 # 梯度惩罚权重
  batch_size: 4                 # 批量大小
  num_epochs: 200               # 训练轮数
```

## 📊 训练监控

### TensorBoard

```bash
tensorboard --logdir logs/sparse_gan
```

查看指标：
- 判别器损失 (`train/discriminator/d_loss`)
- 生成器损失 (`train/generator/g_loss`)
- Wasserstein距离 (`train/discriminator/wasserstein_distance`)
- 梯度惩罚 (`train/discriminator/gradient_penalty`)

### 输出目录结构

```
outputs/sparse_gan/
├── samples/                    # 生成的样本
│   ├── epoch_10/
│   ├── epoch_20/
│   └── ...
├── checkpoints/                # 模型检查点
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_latest.pth
│   └── checkpoint_best.pth
└── logs/                       # TensorBoard日志
```

## 🔧 高级用法

### Python API

```python
import torch
from sparse_gan_models import SparseGenerator, SparseDiscriminator
from sparse_gan_sampling import create_sampler_from_checkpoint

# 加载模型
sampler = create_sampler_from_checkpoint('checkpoints/checkpoint_best.pth')

# 随机采样
voxels = sampler.sample(num_samples=5)

# 从种子采样（可复现）
voxel = sampler.sample_from_seed(seed=42)

# 潜在空间插值
z1 = torch.randn(256)
z2 = torch.randn(256)
interp_voxels = sampler.interpolate(z1, z2, num_steps=10, method='slerp')

# 保存为 NIfTI
sampler.save_as_nifti(voxels[0], 'output.nii.gz')
```

### 自定义数据稀疏化

```python
from sparse_gan_dataset import dense_to_sparse

# 将密集体素转换为稀疏表示
coords, feats = dense_to_sparse(
    dense_voxel,
    threshold=0.1,      # 体素值阈值
    min_voxels=100      # 最少保留的体素数
)
```

## 📝 论文和引用

本项目基于以下工作：

1. **MinkowskiEngine**
   - Choy et al. "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks", CVPR 2019

2. **WGAN-GP**
   - Gulrajani et al. "Improved Training of Wasserstein GANs", NeurIPS 2017

3. **Sparse VAE**
   - MinkowskiEngine VAE 示例

## 🐛 常见问题

### Q: 训练时显存不足？

A: 尝试：
- 减小 `batch_size`
- 减小 `voxel_size`（分辨率）
- 增大 `threshold`（更稀疏）

### Q: 生成的体素质量不好？

A: 尝试：
- 增加训练轮数
- 调整 `n_critic`（判别器训练频率）
- 尝试不同的 GAN 类型
- 检查数据质量和预处理

### Q: 训练不稳定？

A: 
- 使用 `wgan-gp`（最稳定）
- 降低学习率
- 增大 `gradient_penalty_weight`

## 📧 联系和支持

如有问题或建议，请：
- 提交 GitHub Issue
- 查看 MinkowskiEngine 文档：https://github.com/NVIDIA/MinkowskiEngine

## 📄 许可证

本项目遵循 MIT 许可证。

---

**Happy Generating! 🎉**

