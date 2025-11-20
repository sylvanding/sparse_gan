#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinkowskiEngine 稀疏卷积安装验证脚本
测试稀疏卷积的基本功能
"""

import torch
import numpy as np

print("=" * 60)
print("MinkowskiEngine 稀疏卷积安装验证")
print("=" * 60)

# 步骤 1: 导入 MinkowskiEngine
try:
    import MinkowskiEngine as ME
    print("✓ MinkowskiEngine 导入成功")
    print(f"  版本: {ME.__version__}")
except ImportError as e:
    print("✗ MinkowskiEngine 导入失败")
    print(f"  错误: {e}")
    exit(1)

# 步骤 2: 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ 使用设备: {device}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")

# 步骤 3: 创建稀疏张量
print("\n" + "-" * 60)
print("测试 1: 创建稀疏张量")
print("-" * 60)
try:
    # 创建坐标 (batch_index, x, y, z)
    coordinates = torch.IntTensor([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
    ])
    
    # 创建特征 (N, in_channels)
    features = torch.FloatTensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
    ])
    
    # 创建稀疏张量
    sparse_tensor = ME.SparseTensor(
        features=features,
        coordinates=coordinates,
        device=device
    )
    
    print(f"✓ 稀疏张量创建成功")
    print(f"  输入特征维度: {sparse_tensor.F.shape}")
    print(f"  坐标维度: {sparse_tensor.C.shape}")
    print(f"  批次大小: {len(sparse_tensor.decomposed_coordinates)}")
except Exception as e:
    print(f"✗ 稀疏张量创建失败: {e}")
    exit(1)

# 步骤 4: 测试稀疏卷积
print("\n" + "-" * 60)
print("测试 2: 稀疏卷积操作")
print("-" * 60)
try:
    # 创建 3D 稀疏卷积层
    in_channels = 3
    out_channels = 16
    conv = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        dimension=3
    ).to(device)
    
    print(f"✓ 稀疏卷积层创建成功")
    print(f"  输入通道: {in_channels}")
    print(f"  输出通道: {out_channels}")
    print(f"  卷积核大小: 3")
    
    # 前向传播
    output = conv(sparse_tensor)
    print(f"✓ 前向传播成功")
    print(f"  输出特征维度: {output.F.shape}")
    print(f"  输出坐标维度: {output.C.shape}")
except Exception as e:
    print(f"✗ 稀疏卷积操作失败: {e}")
    exit(1)

# 步骤 5: 测试其他常用层
print("\n" + "-" * 60)
print("测试 3: 其他稀疏层")
print("-" * 60)
try:
    # 批归一化
    bn = ME.MinkowskiBatchNorm(out_channels).to(device)
    output = bn(output)
    print("✓ MinkowskiBatchNorm 测试通过")
    
    # ReLU 激活
    relu = ME.MinkowskiReLU()
    output = relu(output)
    print("✓ MinkowskiReLU 测试通过")
    
    # 最大池化
    pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3).to(device)
    output = pool(output)
    print(f"✓ MinkowskiMaxPooling 测试通过")
    print(f"  池化后特征维度: {output.F.shape}")
    
    # 全局池化
    glob_pool = ME.MinkowskiGlobalMaxPooling()
    output = glob_pool(output)
    print(f"✓ MinkowskiGlobalMaxPooling 测试通过")
    print(f"  全局池化后特征维度: {output.F.shape}")
    
except Exception as e:
    print(f"✗ 其他层测试失败: {e}")
    exit(1)

# 步骤 6: 测试梯度反向传播
print("\n" + "-" * 60)
print("测试 4: 梯度反向传播")
print("-" * 60)
try:
    # 创建一个简单的网络
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = ME.MinkowskiConvolution(3, 16, kernel_size=3, dimension=3)
            self.relu = ME.MinkowskiReLU()
            self.conv2 = ME.MinkowskiConvolution(16, 32, kernel_size=3, dimension=3)
            self.pool = ME.MinkowskiGlobalMaxPooling()
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.pool(x)
            return x
    
    net = SimpleNet().to(device)
    
    # 前向传播
    sparse_tensor = ME.SparseTensor(
        features=features,
        coordinates=coordinates,
        device=device
    )
    output = net(sparse_tensor)
    
    # 计算损失并反向传播
    loss = output.F.sum()
    loss.backward()
    
    print("✓ 梯度反向传播成功")
    print(f"  损失值: {loss.item():.4f}")
    
    # 检查梯度
    has_grad = any(p.grad is not None for p in net.parameters())
    if has_grad:
        print("✓ 梯度计算成功")
    else:
        print("✗ 未检测到梯度")
        
except Exception as e:
    print(f"✗ 梯度反向传播失败: {e}")
    exit(1)

# 总结
print("\n" + "=" * 60)
print("✓✓✓ 所有测试通过！MinkowskiEngine 安装成功！✓✓✓")
print("=" * 60)
print("\n稀疏卷积功能正常，可以开始使用 MinkowskiEngine 进行开发。")

