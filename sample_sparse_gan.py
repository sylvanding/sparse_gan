#!/usr/bin/env python
"""
Sparse GAN 采样脚本

使用方式:
    # 随机采样
    python sample_sparse_gan.py --checkpoint checkpoints/sparse_gan/checkpoint_best.pth --num_samples 10
    
    # 插值采样
    python sample_sparse_gan.py --checkpoint checkpoints/sparse_gan/checkpoint_best.pth --interpolate --num_steps 20
    
    # 可视化
    python sample_sparse_gan.py --checkpoint checkpoints/sparse_gan/checkpoint_best.pth --num_samples 5 --visualize
"""

import argparse
import logging
from pathlib import Path

from sparse_gan_sampling import (
    create_sampler_from_checkpoint,
    visualize_voxel_slices,
    visualize_voxel_3d
)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sparse GAN 采样")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='检查点路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='采样数量'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/samples',
        help='输出目录'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='nifti',
        choices=['nifti', 'numpy', 'both'],
        help='输出格式'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否生成可视化图像'
    )
    parser.add_argument(
        '--interpolate',
        action='store_true',
        help='是否生成插值序列'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=10,
        help='插值步数'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于可复现采样）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Sparse GAN 采样")
    logger.info("="*60)
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"采样数量: {args.num_samples}")
    logger.info(f"输出格式: {args.format}")
    logger.info(f"可视化: {args.visualize}")
    logger.info("="*60 + "\n")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建采样器
    logger.info("加载模型...")
    sampler = create_sampler_from_checkpoint(args.checkpoint)
    
    import torch
    
    if args.interpolate:
        logger.info(f"生成 {args.num_steps} 步插值序列...")
        
        # 随机采样两个潜在向量
        if args.seed is not None:
            torch.manual_seed(args.seed)
        
        z1 = torch.randn(sampler.latent_dim, device=sampler.device)
        z2 = torch.randn(sampler.latent_dim, device=sampler.device)
        
        # 插值
        interp_voxels = sampler.interpolate(z1, z2, num_steps=args.num_steps)
        
        # 保存
        for i, voxel in enumerate(interp_voxels):
            if args.format in ['nifti', 'both']:
                sampler.save_as_nifti(
                    voxel,
                    str(output_dir / f'interp_{i:03d}.nii.gz')
                )
            
            if args.format in ['numpy', 'both']:
                sampler.save_as_numpy(
                    voxel,
                    str(output_dir / f'interp_{i:03d}.npy')
                )
            
            if args.visualize:
                visualize_voxel_slices(
                    voxel,
                    save_path=str(output_dir / f'interp_{i:03d}_slices.png')
                )
        
        logger.info(f"插值序列已保存到: {output_dir}")
    
    else:
        logger.info(f"生成 {args.num_samples} 个随机样本...")
        
        # 设置随机种子
        if args.seed is not None:
            torch.manual_seed(args.seed)
            logger.info(f"使用随机种子: {args.seed}")
        
        # 随机采样
        voxels = sampler.sample(num_samples=args.num_samples)
        
        # 保存
        for i, voxel in enumerate(voxels):
            if args.format in ['nifti', 'both']:
                sampler.save_as_nifti(
                    voxel,
                    str(output_dir / f'sample_{i:03d}.nii.gz')
                )
            
            if args.format in ['numpy', 'both']:
                sampler.save_as_numpy(
                    voxel,
                    str(output_dir / f'sample_{i:03d}.npy')
                )
            
            if args.visualize:
                visualize_voxel_slices(
                    voxel,
                    save_path=str(output_dir / f'sample_{i:03d}_slices.png')
                )
        
        logger.info(f"样本已保存到: {output_dir}")
    
    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()

