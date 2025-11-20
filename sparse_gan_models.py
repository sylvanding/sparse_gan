"""
Sparse GAN 模型定义

使用 MinkowskiEngine 实现的 3D 体素生成器和判别器。
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from typing import List, Optional


class SparseGenerator(nn.Module):
    """
    稀疏生成器 - 从潜在向量生成 3D 稀疏体素
    
    架构：
    1. 全连接层：将潜在向量映射到稀疏特征
    2. 生成式转置卷积：逐步上采样
    3. 使用 MinkowskiGenerativeConvolutionTranspose 进行上采样
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        channels: List[int] = [256, 128, 64, 32, 16],
        output_channels: int = 1,
        initial_tensor_stride: int = 32,
        resolution: int = 64,
        kernel_size: int = 3,
        activation: str = "leaky_relu",
        use_batch_norm: bool = True
    ):
        """
        初始化稀疏生成器
        
        Args:
            latent_dim: 潜在向量维度
            channels: 每层的通道数（从高到低）
            output_channels: 输出通道数
            initial_tensor_stride: 初始稀疏张量的步长（决定初始分辨率）
            resolution: 目标分辨率
            kernel_size: 卷积核大小
            activation: 激活函数类型
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        self.initial_tensor_stride = initial_tensor_stride
        self.resolution = resolution
        
        # 计算初始网格大小
        self.initial_size = resolution // initial_tensor_stride
        
        # 全连接层：潜在向量 -> 初始特征
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, channels[0] * (self.initial_size ** 3)),
            nn.LeakyReLU(0.2) if activation == "leaky_relu" else nn.ReLU()
        )
        
        # 第一个卷积: 将 stride=1 的初始网格转换到 initial_tensor_stride
        self.initial_conv = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=kernel_size,
                stride=initial_tensor_stride,
                dimension=3
            ),
            ME.MinkowskiBatchNorm(channels[0]) if use_batch_norm else nn.Identity(),
            ME.MinkowskiLeakyReLU(0.2) if activation == "leaky_relu" else ME.MinkowskiReLU()
        )
        
        # 构建上采样层
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            block = nn.Sequential(
                # 生成式转置卷积（上采样 2x）
                ME.MinkowskiGenerativeConvolutionTranspose(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=2,
                    stride=2,
                    dimension=3
                ),
                ME.MinkowskiBatchNorm(out_ch) if use_batch_norm else nn.Identity(),
                ME.MinkowskiLeakyReLU(0.2) if activation == "leaky_relu" else ME.MinkowskiReLU(),
                # 额外的卷积层（细化特征）
                ME.MinkowskiConvolution(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dimension=3
                ),
                ME.MinkowskiBatchNorm(out_ch) if use_batch_norm else nn.Identity(),
                ME.MinkowskiLeakyReLU(0.2) if activation == "leaky_relu" else ME.MinkowskiReLU()
            )
            self.blocks.append(block)
        
        # 最后一层：输出层
        self.final_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=channels[-1],
                out_channels=output_channels,
                kernel_size=1,
                bias=True,
                dimension=3
            ),
            ME.MinkowskiSigmoid()  # 输出范围 [0, 1]
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, ME.MinkowskiConvolution):
            ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, ME.MinkowskiBatchNorm):
            nn.init.constant_(m.bn.weight, 1)
            nn.init.constant_(m.bn.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor, return_intermediate: bool = False) -> ME.SparseTensor:
        """
        前向传播
        
        Args:
            z: 潜在向量 [batch_size, latent_dim]
            return_intermediate: 是否返回中间特征
            
        Returns:
            生成的稀疏张量
        """
        batch_size = z.size(0)
        device = z.device
        
        # 1. 全连接层：z -> 初始特征
        h = self.fc(z)  # [batch_size, channels[0] * initial_size^3]
        h = h.view(batch_size, self.channels[0], 
                   self.initial_size, self.initial_size, self.initial_size)
        
        # 2. 创建初始坐标网格
        # 生成规则网格坐标
        coords = []
        for b in range(batch_size):
            for x in range(self.initial_size):
                for y in range(self.initial_size):
                    for z in range(self.initial_size):
                        coords.append([b, x, y, z])
        
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        
        # 3. 将密集特征转换为稀疏特征
        # 重新排列特征以匹配坐标顺序
        feats = h.permute(0, 2, 3, 4, 1).contiguous()  # [B, X, Y, Z, C]
        feats = feats.view(-1, self.channels[0])  # [B*X*Y*Z, C]
        
        # 4. 创建初始稀疏张量
        # 注意: 不设置 tensor_stride 参数，让 MinkowskiEngine 自动推断为 stride=1
        # 然后通过 stride=1 的普通卷积将坐标系转换到目标 stride
        sparse_tensor = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=device
        )
        
        # 5. 通过初始卷积层（将坐标从 stride=1 转换到 initial_tensor_stride）
        sparse_tensor = self.initial_conv(sparse_tensor)
        
        # 6. 通过上采样层
        intermediates = [sparse_tensor] if return_intermediate else []
        
        for block in self.blocks:
            sparse_tensor = block(sparse_tensor)
            if return_intermediate:
                intermediates.append(sparse_tensor)
        
        # 7. 最后一层
        output = self.final_block(sparse_tensor)
        
        if return_intermediate:
            return output, intermediates
        
        return output


class SparseDiscriminator(nn.Module):
    """
    稀疏判别器 - 判断 3D 体素是真实还是生成的
    
    架构：
    1. 多层稀疏卷积：提取层次化特征
    2. 全局池化：汇总空间信息
    3. 全连接层：输出真/假分数
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        channels: List[int] = [16, 32, 64, 128, 256],
        kernel_size: int = 3,
        activation: str = "leaky_relu",
        use_batch_norm: bool = True,
        use_spectral_norm: bool = False
    ):
        """
        初始化稀疏判别器
        
        Args:
            input_channels: 输入通道数
            channels: 每层的通道数（从低到高）
            kernel_size: 卷积核大小
            activation: 激活函数类型
            use_batch_norm: 是否使用BatchNorm
            use_spectral_norm: 是否使用谱归一化
        """
        super().__init__()
        
        self.channels = channels
        
        # 构建下采样层
        self.blocks = nn.ModuleList()
        
        in_ch = input_channels
        for out_ch in channels:
            block = nn.Sequential(
                # 下采样卷积
                ME.MinkowskiConvolution(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    dimension=3
                ),
                ME.MinkowskiBatchNorm(out_ch) if use_batch_norm else nn.Identity(),
                ME.MinkowskiLeakyReLU(0.2) if activation == "leaky_relu" else ME.MinkowskiReLU(),
                # 额外的卷积层（细化特征）
                ME.MinkowskiConvolution(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dimension=3
                ),
                ME.MinkowskiBatchNorm(out_ch) if use_batch_norm else nn.Identity(),
                ME.MinkowskiLeakyReLU(0.2) if activation == "leaky_relu" else ME.MinkowskiReLU()
            )
            self.blocks.append(block)
            in_ch = out_ch
        
        # 全局池化
        self.global_pool = ME.MinkowskiGlobalMaxPooling()
        
        # 输出层
        self.final = ME.MinkowskiLinear(channels[-1], 1, bias=True)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, ME.MinkowskiConvolution):
            ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, ME.MinkowskiBatchNorm):
            nn.init.constant_(m.bn.weight, 1)
            nn.init.constant_(m.bn.bias, 0)
        elif isinstance(m, ME.MinkowskiLinear):
            nn.init.normal_(m.linear.weight, 0, 0.02)
            if m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
    
    def forward(self, x: ME.SparseTensor, return_features: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入稀疏张量
            return_features: 是否返回中间特征
            
        Returns:
            判别分数 [batch_size, 1]
        """
        features = []
        
        # 通过下采样层
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 输出层
        output = self.final(x)
        
        if return_features:
            return output.F, features
        
        return output.F


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: ME.SparseTensor,
    fake_samples: ME.SparseTensor,
    device: torch.device
) -> torch.Tensor:
    """
    计算 WGAN-GP 的梯度惩罚
    
    Args:
        discriminator: 判别器模型
        real_samples: 真实样本
        fake_samples: 生成样本
        device: 设备
        
    Returns:
        梯度惩罚损失
    """
    # 注意：对于稀疏张量，梯度惩罚的实现需要特殊处理
    # 我们需要在坐标的并集上进行插值
    
    real_max_idx = real_samples.C[:, 0].max().item() if len(real_samples.C) > 0 else -1
    fake_max_idx = fake_samples.C[:, 0].max().item() if len(fake_samples.C) > 0 else -1
    batch_size = max(real_max_idx, fake_max_idx) + 1
    
    # 1. 准备 alpha
    alpha = torch.rand(batch_size, 1, device=device)
    
    # 2. 处理真实样本
    real_coords = real_samples.C
    real_feats = real_samples.F
    real_batch_indices = real_coords[:, 0].long()
    alpha_real = alpha[real_batch_indices]
    # real_feats 已经是 leaf 或 detached (来自数据加载器)
    weighted_real_feats = real_feats * alpha_real
    
    # 3. 处理生成样本 (需要 detach)
    fake_coords = fake_samples.C
    fake_feats = fake_samples.F.detach()
    fake_batch_indices = fake_coords[:, 0].long()
    alpha_fake = alpha[fake_batch_indices]
    weighted_fake_feats = fake_feats * (1 - alpha_fake)
    
    # 4. 合并坐标和特征
    coords_cat = torch.cat([real_coords, fake_coords], dim=0)
    feats_cat = torch.cat([weighted_real_feats, weighted_fake_feats], dim=0)
    
    # 5. 创建插值稀疏张量
    # 使用 UNWEIGHTED_SUM 模式合并重复坐标处的特征
    # 这实现了: alpha * real + (1-alpha) * fake
    interpolated = ME.SparseTensor(
        features=feats_cat,
        coordinates=coords_cat,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
        device=device
    )
    
    # 6. 设置 requires_grad
    # 我们需要对插值后的特征求导
    interpolated_feats = interpolated.F
    interpolated_feats.requires_grad_(True)
    
    # 更新 interpolated 的 features (确保使用 require_grad 的版本)
    # 注意：虽然 interpolated.F 已经指向这个 tensor，但明确构建一个新的可能更安全，
    # 不过在这里直接使用 interpolated 应该也可以，因为我们修改了 .F 的属性。
    # 为了保险起见，我们构建一个新的 SparseTensor，共享坐标管理器（如果有的话）
    # 但上面的 quantize 操作已经创建了一个新的坐标管理器。
    # 所以我们只需要确保 discriminator 用的是带梯度的 features。
    
    # 重新封装一下以确保万无一失
    interpolated = ME.SparseTensor(
        features=interpolated_feats,
        coordinate_map_key=interpolated.coordinate_map_key,
        coordinate_manager=interpolated.coordinate_manager
    )
    
    # 计算判别器输出
    d_interpolated = discriminator(interpolated)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_feats,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度的 L2 范数
    # 我们需要计算每个样本的梯度范数，而不是每个点的
    # gradients: [N_total, C]
    grad_sq = gradients.pow(2).sum(dim=1)  # [N_total]
    
    # 按批次聚合梯度的平方和
    batch_indices = interpolated.C[:, 0].long()
    # 确保 batch_size 足够大以包含所有索引
    current_batch_size = batch_indices.max().item() + 1
    final_batch_size = max(batch_size, current_batch_size)
    
    grad_sq_sum = torch.zeros(final_batch_size, device=device)
    grad_sq_sum.scatter_add_(0, batch_indices, grad_sq)
    
    # 计算每个样本的梯度范数
    gradients_norm = torch.sqrt(grad_sq_sum + 1e-8)  # [batch_size]
    
    # 计算梯度惩罚（平均所有样本）
    # 只对存在的样本计算（排除可能的空样本，虽然这里应该都有）
    # 如果某个 batch index 没有点，grad_sq_sum 为 0，gradients_norm 为 0 (eps)。
    # 这会导致 (0-1)^2 = 1 的惩罚。这是不合理的，因为空样本没有梯度。
    # 我们应该只对非空样本计算惩罚。
    
    mask = grad_sq_sum > 0
    if mask.sum() > 0:
        gradient_penalty = ((gradients_norm[mask] - 1) ** 2).mean()
    else:
        gradient_penalty = torch.tensor(0.0, device=device, requires_grad=True)
    
    return gradient_penalty


if __name__ == "__main__":
    """测试模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试生成器
    print("测试生成器...")
    generator = SparseGenerator(
        latent_dim=256,
        channels=[256, 128, 64, 32, 16],
        resolution=64
    ).to(device)
    
    z = torch.randn(2, 256, device=device)
    fake_voxels = generator(z)
    print(f"生成器输出: {fake_voxels}")
    print(f"  - 坐标形状: {fake_voxels.C.shape}")
    print(f"  - 特征形状: {fake_voxels.F.shape}")
    
    # 测试判别器
    print("\n测试判别器...")
    discriminator = SparseDiscriminator(
        input_channels=1,
        channels=[16, 32, 64, 128, 256]
    ).to(device)
    
    d_output = discriminator(fake_voxels)
    print(f"判别器输出: {d_output.shape}")
    print(f"  - 输出值: {d_output}")
    
    print("\n模型测试完成！")

