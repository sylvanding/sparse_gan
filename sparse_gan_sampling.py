"""
Sparse GAN 采样和可视化

提供生成、插值、可视化等功能。
"""

import torch
import numpy as np
import MinkowskiEngine as ME
from pathlib import Path
from typing import Optional, List
import logging
import yaml
import nibabel as nib

from sparse_gan_models import SparseGenerator
from sparse_gan_dataset import sparse_to_dense

logger = logging.getLogger(__name__)


class SparseGANSampler:
    """
    Sparse GAN 采样器
    
    支持随机采样、插值、条件生成等功能。
    """
    
    def __init__(
        self,
        generator: SparseGenerator,
        device: torch.device,
        latent_dim: int = 256,
        resolution: int = 64
    ):
        """
        初始化采样器
        
        Args:
            generator: 生成器模型
            device: 设备
            latent_dim: 潜在向量维度
            resolution: 生成分辨率
        """
        self.generator = generator
        self.device = device
        self.latent_dim = latent_dim
        self.resolution = resolution
        
        self.generator.eval()
    
    @torch.no_grad()
    def sample(self, num_samples: int = 1, return_sparse: bool = False):
        """
        随机采样
        
        Args:
            num_samples: 采样数量
            return_sparse: 是否返回稀疏张量
            
        Returns:
            生成的体素（密集或稀疏）
        """
        # 采样潜在向量
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # 生成
        fake_sparse = self.generator(z)
        
        if return_sparse:
            return fake_sparse
        
        # 转换为密集格式
        dense_voxels = []
        for i in range(num_samples):
            dense = sparse_to_dense(fake_sparse, self.resolution, batch_idx=i)
            dense_voxels.append(dense)
        
        return torch.stack(dense_voxels, dim=0)
    
    @torch.no_grad()
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
        method: str = 'linear'
    ):
        """
        在两个潜在向量之间插值
        
        Args:
            z1: 起始潜在向量 [latent_dim]
            z2: 结束潜在向量 [latent_dim]
            num_steps: 插值步数
            method: 插值方法 ('linear' 或 'slerp')
            
        Returns:
            插值序列的密集体素
        """
        if method == 'linear':
            # 线性插值
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            z_interp = torch.stack([
                (1 - alpha) * z1 + alpha * z2
                for alpha in alphas
            ])
        
        elif method == 'slerp':
            # 球面线性插值
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            
            # 归一化
            z1_norm = z1 / z1.norm()
            z2_norm = z2 / z2.norm()
            
            # 计算夹角
            omega = torch.acos((z1_norm * z2_norm).sum())
            
            # Slerp
            z_interp = torch.stack([
                (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * z1 +
                (torch.sin(alpha * omega) / torch.sin(omega)) * z2
                for alpha in alphas
            ])
        
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 生成
        fake_sparse = self.generator(z_interp)
        
        # 转换为密集格式
        dense_voxels = []
        for i in range(num_steps):
            dense = sparse_to_dense(fake_sparse, self.resolution, batch_idx=i)
            dense_voxels.append(dense)
        
        return torch.stack(dense_voxels, dim=0)
    
    @torch.no_grad()
    def sample_from_seed(self, seed: int) -> torch.Tensor:
        """
        从指定种子采样（可复现）
        
        Args:
            seed: 随机种子
            
        Returns:
            生成的密集体素
        """
        # 保存当前随机状态
        rng_state = torch.get_rng_state()
        
        # 设置种子
        torch.manual_seed(seed)
        
        # 采样
        voxel = self.sample(num_samples=1)
        
        # 恢复随机状态
        torch.set_rng_state(rng_state)
        
        return voxel[0]
    
    def save_as_nifti(
        self,
        voxel: torch.Tensor,
        save_path: str,
        affine: Optional[np.ndarray] = None
    ):
        """
        保存为 NIfTI 格式
        
        Args:
            voxel: 体素数据 [C, X, Y, Z]
            save_path: 保存路径
            affine: 仿射矩阵
        """
        # 转换为 numpy
        voxel_np = voxel.cpu().numpy()
        
        # 如果有多个通道，只保存第一个
        if voxel_np.shape[0] > 1:
            voxel_np = voxel_np[0]
        else:
            voxel_np = voxel_np.squeeze(0)
        
        # 创建默认仿射矩阵
        if affine is None:
            affine = np.eye(4)
        
        # 创建 NIfTI 图像
        nifti_img = nib.Nifti1Image(voxel_np, affine)
        
        # 保存
        nib.save(nifti_img, save_path)
        logger.info(f"保存 NIfTI: {save_path}")
    
    def save_as_numpy(self, voxel: torch.Tensor, save_path: str):
        """
        保存为 NumPy 格式
        
        Args:
            voxel: 体素数据 [C, X, Y, Z]
            save_path: 保存路径
        """
        voxel_np = voxel.cpu().numpy()
        np.save(save_path, voxel_np)
        logger.info(f"保存 NumPy: {save_path}")


def visualize_voxel_3d(
    voxel: torch.Tensor,
    threshold: float = 0.0,
    save_path: Optional[str] = None
):
    """
    3D 可视化体素（使用 matplotlib）
    
    Args:
        voxel: 体素数据 [C, X, Y, Z]
        threshold: 显示阈值
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib 未安装，无法可视化")
        return
    
    # 提取第一个通道
    if voxel.dim() == 4:
        voxel = voxel[0]
    
    voxel_np = voxel.cpu().numpy()
    
    # 找到非零体素
    mask = voxel_np > threshold
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        logger.warning("没有体素超过阈值")
        return
    
    # 创建 3D 图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制体素
    values = voxel_np[mask]
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=values,
        cmap='viridis',
        marker='s',
        s=20,
        alpha=0.6
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Voxel (体素数: {len(coords)})')
    
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"保存可视化: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_voxel_slices(
    voxel: torch.Tensor,
    num_slices: int = 5,
    save_path: Optional[str] = None
):
    """
    可视化体素切片
    
    Args:
        voxel: 体素数据 [C, X, Y, Z]
        num_slices: 每个轴的切片数量
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，无法可视化")
        return
    
    # 提取第一个通道
    if voxel.dim() == 4:
        voxel = voxel[0]
    
    voxel_np = voxel.cpu().numpy()
    X, Y, Z = voxel_np.shape
    
    # 创建子图
    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices * 3, 9))
    
    # X 轴切片
    for i, idx in enumerate(np.linspace(0, X-1, num_slices, dtype=int)):
        axes[0, i].imshow(voxel_np[idx, :, :], cmap='gray')
        axes[0, i].set_title(f'X={idx}')
        axes[0, i].axis('off')
    
    # Y 轴切片
    for i, idx in enumerate(np.linspace(0, Y-1, num_slices, dtype=int)):
        axes[1, i].imshow(voxel_np[:, idx, :], cmap='gray')
        axes[1, i].set_title(f'Y={idx}')
        axes[1, i].axis('off')
    
    # Z 轴切片
    for i, idx in enumerate(np.linspace(0, Z-1, num_slices, dtype=int)):
        axes[2, i].imshow(voxel_np[:, :, idx], cmap='gray')
        axes[2, i].set_title(f'Z={idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"保存切片可视化: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_sampler_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None
) -> SparseGANSampler:
    """
    从检查点创建采样器
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        采样器实例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 创建生成器
    gen_config = config['generator']
    generator = SparseGenerator(
        latent_dim=gen_config['latent_dim'],
        channels=gen_config['channels'],
        output_channels=gen_config['output_channels'],
        initial_tensor_stride=gen_config['initial_tensor_stride'],
        resolution=config['data']['voxel_size'],
        kernel_size=gen_config['kernel_size'],
        activation=gen_config['activation'],
        use_batch_norm=gen_config['use_batch_norm']
    ).to(device)
    
    # 加载权重
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # 创建采样器
    sampler = SparseGANSampler(
        generator=generator,
        device=device,
        latent_dim=gen_config['latent_dim'],
        resolution=config['data']['voxel_size']
    )
    
    logger.info(f"从检查点创建采样器: {checkpoint_path}")
    logger.info(f"  - Epoch: {checkpoint['epoch']}")
    logger.info(f"  - 分辨率: {config['data']['voxel_size']}")
    
    return sampler


if __name__ == "__main__":
    """采样脚本"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Sparse GAN 采样")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='检查点路径')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='采样数量')
    parser.add_argument('--output_dir', type=str, default='./outputs/samples',
                        help='输出目录')
    parser.add_argument('--format', type=str, default='nifti',
                        choices=['nifti', 'numpy', 'both'],
                        help='输出格式')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化')
    parser.add_argument('--interpolate', action='store_true',
                        help='是否生成插值序列')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建采样器
    sampler = create_sampler_from_checkpoint(args.checkpoint)
    
    if args.interpolate:
        logger.info("生成插值序列...")
        
        # 随机采样两个潜在向量
        z1 = torch.randn(sampler.latent_dim, device=sampler.device)
        z2 = torch.randn(sampler.latent_dim, device=sampler.device)
        
        # 插值
        interp_voxels = sampler.interpolate(z1, z2, num_steps=args.num_samples)
        
        # 保存
        for i, voxel in enumerate(interp_voxels):
            if args.format in ['nifti', 'both']:
                sampler.save_as_nifti(voxel, str(output_dir / f'interp_{i:03d}.nii.gz'))
            
            if args.format in ['numpy', 'both']:
                sampler.save_as_numpy(voxel, str(output_dir / f'interp_{i:03d}.npy'))
            
            if args.visualize:
                visualize_voxel_slices(
                    voxel,
                    save_path=str(output_dir / f'interp_{i:03d}_slices.png')
                )
    
    else:
        logger.info(f"生成 {args.num_samples} 个随机样本...")
        
        # 随机采样
        voxels = sampler.sample(num_samples=args.num_samples)
        
        # 保存
        for i, voxel in enumerate(voxels):
            if args.format in ['nifti', 'both']:
                sampler.save_as_nifti(voxel, str(output_dir / f'sample_{i:03d}.nii.gz'))
            
            if args.format in ['numpy', 'both']:
                sampler.save_as_numpy(voxel, str(output_dir / f'sample_{i:03d}.npy'))
            
            if args.visualize:
                visualize_voxel_slices(
                    voxel,
                    save_path=str(output_dir / f'sample_{i:03d}_slices.png')
                )
    
    logger.info(f"完成！样本保存到: {output_dir}")

