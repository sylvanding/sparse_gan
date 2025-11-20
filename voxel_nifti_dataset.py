"""
NIfTI体素数据集

用于加载NIfTI格式的体素数据，支持自适应分辨率调整和数据增强。
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# 添加GenerativeModels到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "GenerativeModels"))

import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset, CacheDataset
from monai.utils import set_determinism
from monai.config import KeysCollection
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


class InvertIntensityd(MapTransform):
    """
    亮度反转变换
    
    将图像中最小的亮度值变成最大的，最大的亮度值变成最小的。
    公式: output = max_value - input + min_value
    
    注意: 此变换应该在归一化之后应用，此时图像值在[0, 1]或[-1, 1]范围内。
    """
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        """
        初始化亮度反转变换
        
        Args:
            keys: 要处理的数据字典键
            allow_missing_keys: 是否允许缺失键
        """
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        """
        应用亮度反转
        
        Args:
            data: 输入数据字典
            
        Returns:
            转换后的数据字典
        """
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            
            # 获取当前图像的最小值和最大值
            min_val = torch.min(img)
            max_val = torch.max(img)
            
            # 反转: output = max - input + min
            # 这样原来的min会变成max，原来的max会变成min
            d[key] = max_val - img + min_val
        
        return d


class VoxelNiftiDataset:
    """
    NIfTI体素数据集包装器
    
    自动加载NIfTI文件并应用MONAI transforms进行预处理。
    """
    
    def __init__(
        self,
        data_dir: str,
        target_voxel_size = 64,
        voxel_resize = None,
        normalize_to_minus_one_one: bool = False,
        cache_rate: float = 0.0,
        cache_num_workers: int = 4,
        augmentation: bool = False,
        augmentation_config: Optional[Dict] = None,
        max_data_size: Optional[int] = None,
    ):
        """
        初始化NIfTI数据集
        
        Args:
            data_dir: NIfTI文件目录
            target_voxel_size: 目标体素分辨率，可以是整数（各向同性）或[X, Y, Z]列表（各向异性）
            voxel_resize: 预处理resize分辨率，在裁剪前先resize到此分辨率，None表示不进行预resize
            normalize_to_minus_one_one: 是否归一化到[-1, 1]
            cache_rate: 缓存比例 (0-1)
            cache_num_workers: CacheDataset的工作进程数
            augmentation: 是否启用数据增强
            augmentation_config: 数据增强配置
            max_data_size: 最大数据大小，None表示不限制
        """
        self.data_dir = Path(data_dir)
        
        # 处理target_voxel_size，支持整数或列表
        if isinstance(target_voxel_size, (list, tuple)):
            if len(target_voxel_size) != 3:
                raise ValueError(f"target_voxel_size作为列表时必须包含3个元素[X, Y, Z]，但得到了{len(target_voxel_size)}个元素")
            self.target_voxel_size = tuple(target_voxel_size)
        else:
            self.target_voxel_size = (target_voxel_size, target_voxel_size, target_voxel_size)
        
        # 处理voxel_resize，支持整数或列表
        if voxel_resize is not None:
            if isinstance(voxel_resize, (list, tuple)):
                if len(voxel_resize) != 3:
                    raise ValueError(f"voxel_resize作为列表时必须包含3个元素[X, Y, Z]，但得到了{len(voxel_resize)}个元素")
                self.voxel_resize = tuple(voxel_resize)
            else:
                self.voxel_resize = (voxel_resize, voxel_resize, voxel_resize)
        else:
            self.voxel_resize = None
        
        self.normalize_to_minus_one_one = normalize_to_minus_one_one
        self.cache_rate = cache_rate
        self.cache_num_workers = cache_num_workers
        self.augmentation = augmentation
        
        if augmentation_config is None:
            augmentation_config = {
                'random_flip_prob': 0.5,
                # 'random_rotate_prob': 0.5
            }
        self.augmentation_config = augmentation_config
        
        # 收集所有NIfTI文件
        self.nifti_files = self._collect_nifti_files()
        if max_data_size is not None and len(self.nifti_files) > max_data_size:
            self.nifti_files = self.nifti_files[:max_data_size]
            logger.info(f"限制数据集大小为 {max_data_size} 个样本")
        
        logger.info(f"初始化NIfTI数据集:")
        logger.info(f"  数据目录: {data_dir}")
        logger.info(f"  样本数量: {len(self.nifti_files)}")
        logger.info(f"  目标体素大小: {self.target_voxel_size[0]}x{self.target_voxel_size[1]}x{self.target_voxel_size[2]}")
        if self.voxel_resize is not None:
            logger.info(f"  预处理resize: {self.voxel_resize[0]}x{self.voxel_resize[1]}x{self.voxel_resize[2]}")
        else:
            logger.info(f"  预处理resize: 不进行预resize")
        logger.info(f"  缓存比例: {cache_rate}")
        logger.info(f"  数据增强: {augmentation}")
        
        # 创建transforms
        self.transforms = self._create_transforms()
        
        # 创建数据列表
        self.data_list = [{"image": str(f)} for f in self.nifti_files]
        
        # 创建MONAI Dataset
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_list,
                transform=self.transforms,
                cache_rate=cache_rate,
                num_workers=cache_num_workers
            )
            logger.info(f"使用缓存数据集 (cache_rate={cache_rate}, num_workers={cache_num_workers})")
        else:
            self.dataset = Dataset(
                data=self.data_list,
                transform=self.transforms
            )
            logger.info("使用标准数据集 (无缓存)")
    
    def _collect_nifti_files(self) -> List[Path]:
        """收集所有NIfTI文件"""
        nifti_files = []
        
        # 支持.nii和.nii.gz
        for pattern in ['*.nii', '*.nii.gz']:
            nifti_files.extend(self.data_dir.glob(pattern))
        
        nifti_files.sort()  # 确保顺序一致
        
        if len(nifti_files) == 0:
            raise ValueError(f"在目录 {self.data_dir} 中未找到NIfTI文件")
        
        return nifti_files
    
    def _create_transforms(self):
        """
        创建MONAI transforms管道
        
        ⭐ 支持Patch-Based训练策略：
        - 训练时：使用RandSpatialCropd随机裁剪patch，显著提升训练速度
        - 验证时：使用CenterSpatialCropd中心裁剪或直接resize
        """
        transform_list = [
            # 加载NIfTI文件
            transforms.LoadImaged(keys=["image"]),
            
            # 确保通道维度在前
            transforms.EnsureChannelFirstd(keys=["image"]),
            
            # 调整空间分辨率（统一体素spacing）
            transforms.Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear"
            ),
        ]
        
        # ⭐ 预处理resize（如果配置了voxel_resize）
        if self.voxel_resize is not None:
            logger.info(f"  添加预处理resize变换: {self.voxel_resize}")
            transform_list.append(
                transforms.Resized(
                    keys=["image"],
                    spatial_size=self.voxel_resize,
                    mode="trilinear",
                    align_corners=False
                )
            )
        
        # ⭐⭐⭐ Patch-Based训练策略
        if self.augmentation:
            # 训练集
            
            # 检查是否需要使用patch-based训练
            use_patch_based = self.augmentation_config.get('use_patch_based', True)
            
            if use_patch_based:
                logger.info(f"  训练模式: 使用Patch-Based训练 (随机裁剪patch)")
                
                # 方案1：如果原始数据可能小于目标大小，先padding
                transform_list.append(
                    transforms.SpatialPadd(
                        keys=["image"],
                        spatial_size=self.target_voxel_size,
                        mode="constant"
                    )
                )
                
                # ⭐ 关键：随机裁剪patch（每个epoch看到不同的区域）
                transform_list.append(
                    transforms.RandSpatialCropd(
                        keys=["image"],
                        roi_size=self.target_voxel_size,
                        random_size=False
                    )
                )
            else:
                # 传统方案：直接resize（速度慢，不推荐）
                logger.warning("  未启用Patch-Based训练，将直接resize（速度较慢）")
            
            # 数据增强（在裁剪前进行）
            # 随机翻转
            if self.augmentation_config.get('random_flip_prob', 0) > 0:
                transform_list.append(
                    transforms.RandFlipd(
                        keys=["image"],
                        prob=self.augmentation_config['random_flip_prob'],
                        spatial_axis=[0, 1, 2]
                    )
                )
            
            # 随机旋转
            if self.augmentation_config.get('random_rotate_prob', 0) > 0:
                transform_list.append(
                    transforms.RandRotate90d(
                        keys=["image"],
                        prob=self.augmentation_config['random_rotate_prob'],
                        spatial_axes=(0, 1)
                    )
                )
                
            # 随机裁剪
            transform_list.append(
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=self.target_voxel_size,
                )
            )
        else:
            # 验证集：使用中心裁剪或直接resize
            logger.info(f"  验证模式: 使用中心裁剪或resize")
            use_center_crop = self.augmentation_config.get('use_center_crop_for_val', True)
            
            if use_center_crop:
                # 方案1：中心裁剪（推荐，与训练保持一致）
                # transform_list.append(
                #     transforms.SpatialPadd(
                #         keys=["image"],
                #         spatial_size=self.target_voxel_size,
                #         mode="constant"
                #     )
                # )
                transform_list.append(
                    transforms.CenterSpatialCropd(
                        keys=["image"],
                        roi_size=self.target_voxel_size
                    )
                )
            else:
                # 方案2：直接resize
                transform_list.append(
                    transforms.Resized(
                        keys=["image"],
                        spatial_size=self.target_voxel_size,
                        mode="trilinear",
                        align_corners=False
                    )
                )
        
        # 归一化
        if self.normalize_to_minus_one_one:
            # 归一化到[-1, 1]
            transform_list.append(
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=0.0,
                    a_max=255.0,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True
                )
            )
        else:
            # 归一化到[0, 1]
            transform_list.append(
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=0.0,
                    a_max=255.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                )
            )
        
        # ⭐ 亮度反转（在归一化之后应用）
        if self.augmentation_config.get('invert_intensity', False):
            logger.info("  启用亮度反转增强")
            transform_list.append(
                InvertIntensityd(keys=["image"])
            )
        
        # 确保类型为float32
        transform_list.append(
            transforms.EnsureTyped(keys=["image"], dtype=torch.float32)
        )
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.dataset[idx]


def create_train_val_dataloaders(
    config: Dict[str, Any],
    train_data_dir: Optional[str] = None,
    val_data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """
    创建训练和验证DataLoader
    
    Args:
        config: 配置字典
        train_data_dir: 训练数据目录（覆盖config）
        val_data_dir: 验证数据目录（覆盖config）
        batch_size: 批量大小（覆盖config）
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    # 提取配置
    data_config = config['data']
    
    train_dir = train_data_dir or data_config['train_data_dir']
    val_dir = val_data_dir or data_config['val_data_dir']
    
    # 处理voxel_size，支持整数或列表
    voxel_size = data_config['voxel_size']
    if isinstance(voxel_size, (list, tuple)):
        if len(voxel_size) != 3:
            raise ValueError(f"voxel_size作为列表时必须包含3个元素[X, Y, Z]，但得到了{len(voxel_size)}个元素")
        voxel_size_tuple = tuple(voxel_size)
    else:
        voxel_size_tuple = voxel_size
    
    # 处理voxel_resize，支持整数或列表或None
    voxel_resize = data_config.get('voxel_resize', None)
    if voxel_resize is not None:
        if isinstance(voxel_resize, (list, tuple)):
            if len(voxel_resize) != 3:
                raise ValueError(f"voxel_resize作为列表时必须包含3个元素[X, Y, Z]，但得到了{len(voxel_resize)}个元素")
            voxel_resize_tuple = tuple(voxel_resize)
        else:
            voxel_resize_tuple = voxel_resize
    else:
        voxel_resize_tuple = None
    
    cache_rate = data_config.get('cache_rate', 0.0)
    num_workers = data_config.get('num_workers', 2)
    pin_memory = data_config.get('pin_memory', True)
    
    augmentation_config = data_config.get('augmentation', {})
    augmentation_enabled = augmentation_config.get('enabled', False)
    
    # 获取batch_size（从vqvae、autoencoder或diffusion配置中）
    if batch_size is None:
        if 'vqvae' in config and 'training' in config['vqvae']:
            batch_size = config['vqvae']['training']['batch_size']
        elif 'autoencoder' in config and 'training' in config['autoencoder']:
            batch_size = config['autoencoder']['training']['batch_size']
        elif 'diffusion' in config and 'training' in config['diffusion']:
            batch_size = config['diffusion']['training']['batch_size']
        else:
            batch_size = 2  # 默认值
    
    logger.info("创建训练和验证数据集...")
    
    max_data_size_train = data_config.get('max_data_size_for_train', None)
    max_data_size_val = data_config.get('max_data_size_for_val', None)
    
    # 创建训练数据集（启用数据增强）
    train_dataset = VoxelNiftiDataset(
        data_dir=train_dir,
        target_voxel_size=voxel_size_tuple,
        voxel_resize=voxel_resize_tuple,
        normalize_to_minus_one_one=False,
        cache_rate=cache_rate,
        cache_num_workers=num_workers,
        augmentation=augmentation_enabled,
        augmentation_config=augmentation_config,
        max_data_size=max_data_size_train
    )
    
    # 创建验证数据集（不使用数据增强）
    val_dataset = VoxelNiftiDataset(
        data_dir=val_dir,
        target_voxel_size=voxel_size_tuple,
        voxel_resize=voxel_resize_tuple,
        normalize_to_minus_one_one=False,
        cache_rate=cache_rate,
        cache_num_workers=num_workers,
        augmentation=False,
        augmentation_config=augmentation_config,
        max_data_size=max_data_size_val
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"Batch大小: {batch_size}")
    
    return train_loader, val_loader

