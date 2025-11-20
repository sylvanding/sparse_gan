"""
Sparse GAN 数据集

加载 NIfTI 体素数据并转换为稀疏表示。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
from typing import Dict, Tuple, Optional
import logging

from voxel_nifti_dataset import VoxelNiftiDataset, create_train_val_dataloaders

logger = logging.getLogger(__name__)


def dense_to_sparse(
    dense_voxel: torch.Tensor,
    threshold: float = 0.1,
    min_voxels: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将密集体素转换为稀疏表示
    
    Args:
        dense_voxel: 密集体素 [C, X, Y, Z]
        threshold: 体素值阈值，大于此值的体素会被保留
        min_voxels: 最少保留的体素数
        
    Returns:
        coords: 坐标 [N, 3] (X, Y, Z)
        feats: 特征 [N, C]
    """
    # 如果输入是 [1, C, X, Y, Z]，去掉 batch 维度
    if dense_voxel.dim() == 5:
        dense_voxel = dense_voxel.squeeze(0)
    
    C, X, Y, Z = dense_voxel.shape
    
    # 方法1：基于阈值的稀疏化
    # 计算每个体素的强度（对所有通道求和）
    if C > 1:
        intensity = dense_voxel.sum(dim=0)  # [X, Y, Z]
    else:
        intensity = dense_voxel.squeeze(0)  # [X, Y, Z]
    
    # 找到大于阈值的体素
    mask = intensity > threshold
    
    # 如果稀疏体素太少，降低阈值或选择 top-k
    num_voxels = mask.sum().item()
    if num_voxels < min_voxels:
        # 选择强度最大的 min_voxels 个体素
        flat_intensity = intensity.view(-1)
        top_k = min(min_voxels, flat_intensity.numel())
        _, top_indices = torch.topk(flat_intensity, top_k)
        
        # 创建新的mask
        mask = torch.zeros_like(flat_intensity, dtype=torch.bool)
        mask[top_indices] = True
        mask = mask.view(X, Y, Z)
    
    # 获取非零体素的坐标
    coords = mask.nonzero(as_tuple=False)  # [N, 3]
    
    # 获取对应的特征值
    feats = dense_voxel[:, mask].T  # [N, C]
    
    return coords, feats


def collate_sparse_gan_batch(batch_list):
    """
    批量整理函数：将多个样本整理成稀疏张量批次
    
    Args:
        batch_list: 列表，每个元素是一个字典 {"image": tensor}
        
    Returns:
        字典，包含稀疏张量和原始密集数据
    """
    # 提取所有图像
    images = [item["image"] for item in batch_list]
    batch_size = len(images)
    
    # 转换为密集张量批次
    dense_batch = torch.stack(images, dim=0)  # [B, C, X, Y, Z]
    
    # 转换为稀疏表示
    all_coords = []
    all_feats = []
    
    for batch_idx, dense_voxel in enumerate(dense_batch):
        coords, feats = dense_to_sparse(dense_voxel, threshold=0.1)
        
        # 添加 batch 索引
        batch_coords = torch.cat([
            torch.full((coords.shape[0], 1), batch_idx, dtype=torch.int32),
            coords.int()
        ], dim=1)
        
        all_coords.append(batch_coords)
        all_feats.append(feats)
    
    # 合并所有批次
    batched_coords = torch.cat(all_coords, dim=0)
    batched_feats = torch.cat(all_feats, dim=0)
    
    return {
        "sparse_coords": batched_coords,
        "sparse_feats": batched_feats,
        "dense_voxels": dense_batch
    }


def create_sparse_tensor_from_batch(
    batch_dict: Dict,
    device: torch.device
) -> ME.SparseTensor:
    """
    从批次字典创建稀疏张量
    
    Args:
        batch_dict: 包含 sparse_coords 和 sparse_feats 的字典
        device: 目标设备
        
    Returns:
        MinkowskiEngine 稀疏张量
    """
    coords = batch_dict["sparse_coords"].to(device)
    feats = batch_dict["sparse_feats"].to(device)
    
    sparse_tensor = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        device=device
    )
    
    return sparse_tensor


class SparseGANDataModule:
    """
    Sparse GAN 数据模块
    
    封装数据加载、预处理和批次整理的完整流程。
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据模块
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.training_config = config['training']
        
        # 创建数据集
        self.train_loader = None
        self.val_loader = None
        
    def setup(self):
        """设置数据加载器"""
        logger.info("设置 Sparse GAN 数据加载器...")
        
        # 使用 voxel_nifti_dataset 创建基础数据加载器
        train_loader, val_loader = create_train_val_dataloaders(
            config=self.config,
            batch_size=self.training_config['batch_size']
        )
        
        # 包装成稀疏数据加载器
        self.train_loader = self._wrap_dataloader(train_loader)
        self.val_loader = self._wrap_dataloader(val_loader)
        
        logger.info(f"训练批次数: {len(self.train_loader)}")
        logger.info(f"验证批次数: {len(self.val_loader)}")
    
    def _wrap_dataloader(self, base_loader):
        """
        包装基础数据加载器，添加稀疏转换
        
        Args:
            base_loader: 基础数据加载器
            
        Returns:
            包装后的数据加载器
        """
        # 创建新的数据加载器，使用自定义的 collate 函数
        wrapped_loader = DataLoader(
            dataset=base_loader.dataset,
            batch_size=base_loader.batch_size,
            shuffle=isinstance(base_loader.sampler, torch.utils.data.RandomSampler),
            num_workers=base_loader.num_workers,
            pin_memory=base_loader.pin_memory,
            collate_fn=collate_sparse_gan_batch,
            persistent_workers=base_loader.persistent_workers if base_loader.num_workers > 0 else False
        )
        
        return wrapped_loader
    
    def get_train_dataloader(self):
        """获取训练数据加载器"""
        return self.train_loader
    
    def get_val_dataloader(self):
        """获取验证数据加载器"""
        return self.val_loader


def visualize_sparse_tensor(
    sparse_tensor: ME.SparseTensor,
    batch_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    可视化稀疏张量
    
    Args:
        sparse_tensor: 稀疏张量
        batch_idx: 批次索引
        save_path: 保存路径
    """
    # 提取指定批次的坐标和特征
    coords = sparse_tensor.coordinates_at(batch_idx)
    feats = sparse_tensor.features_at(batch_idx)
    
    if len(coords) == 0:
        logger.warning(f"批次 {batch_idx} 没有体素")
        return
    
    logger.info(f"批次 {batch_idx}:")
    logger.info(f"  - 体素数量: {len(coords)}")
    logger.info(f"  - 坐标范围: X[{coords[:, 0].min()}, {coords[:, 0].max()}], "
                f"Y[{coords[:, 1].min()}, {coords[:, 1].max()}], "
                f"Z[{coords[:, 2].min()}, {coords[:, 2].max()}]")
    logger.info(f"  - 特征范围: [{feats.min():.3f}, {feats.max():.3f}]")


def sparse_to_dense(
    sparse_tensor: ME.SparseTensor,
    resolution: int = 64,
    batch_idx: int = 0
) -> torch.Tensor:
    """
    将稀疏张量转换回密集体素（用于可视化）
    
    Args:
        sparse_tensor: 稀疏张量
        resolution: 目标分辨率
        batch_idx: 批次索引
        
    Returns:
        密集体素 [C, X, Y, Z]
    """
    coords = sparse_tensor.coordinates_at(batch_idx)
    feats = sparse_tensor.features_at(batch_idx)
    
    if len(coords) == 0:
        # 返回空体素
        num_channels = sparse_tensor.F.shape[1]
        return torch.zeros(num_channels, resolution, resolution, resolution)
    
    num_channels = feats.shape[1]
    dense_voxel = torch.zeros(
        num_channels, resolution, resolution, resolution,
        device=feats.device
    )
    
    # 填充非零体素
    # 注意：coords 的格式是 [batch_idx, x, y, z]，需要去掉 batch_idx
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # 确保坐标在有效范围内
    valid_mask = (x >= 0) & (x < resolution) & \
                 (y >= 0) & (y < resolution) & \
                 (z >= 0) & (z < resolution)
    
    x = x[valid_mask].long()
    y = y[valid_mask].long()
    z = z[valid_mask].long()
    feats = feats[valid_mask]
    
    # 填充体素
    for c in range(num_channels):
        dense_voxel[c, x, y, z] = feats[:, c]
    
    return dense_voxel


if __name__ == "__main__":
    """测试数据加载"""
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # 加载配置
    with open("sparse_gan_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 修改配置用于测试
    config['data']['train_data_dir'] = "/data/nifti/train"
    config['data']['val_data_dir'] = "/data/nifti/val"
    config['training']['batch_size'] = 2
    
    # 创建数据模块
    data_module = SparseGANDataModule(config)
    
    try:
        data_module.setup()
        
        # 测试训练数据加载
        print("\n测试训练数据加载...")
        train_loader = data_module.get_train_dataloader()
        
        for batch in train_loader:
            print(f"批次键: {batch.keys()}")
            print(f"稀疏坐标形状: {batch['sparse_coords'].shape}")
            print(f"稀疏特征形状: {batch['sparse_feats'].shape}")
            print(f"密集体素形状: {batch['dense_voxels'].shape}")
            
            # 创建稀疏张量
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sparse_tensor = create_sparse_tensor_from_batch(batch, device)
            
            print(f"\n稀疏张量:")
            visualize_sparse_tensor(sparse_tensor, batch_idx=0)
            
            break
        
        print("\n数据加载测试完成！")
    
    except Exception as e:
        print(f"测试失败（可能是数据路径不存在）: {e}")
        print("请修改配置文件中的数据路径后重试。")

