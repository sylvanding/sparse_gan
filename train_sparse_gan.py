#!/usr/bin/env python
"""
Sparse GAN 训练启动脚本

使用方式:
    python train_sparse_gan.py --config sparse_gan_config.yaml
    
恢复训练:
    python train_sparse_gan.py --config sparse_gan_config.yaml --resume checkpoints/sparse_gan/checkpoint_latest.pth
"""

import os
import sys
import argparse
import yaml
import logging
import shutil
import numpy as np
import torch
from pathlib import Path

from sparse_gan_trainer import SparseGANTrainer
from sparse_gan_dataset import SparseGANDataModule


def clean_directories(config: dict):
    """
    清空检查点和日志目录
    
    Args:
        config: 配置字典
    """
    checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
    log_dir = Path(config['experiment']['log_dir'])
    
    if checkpoint_dir.exists():
        print(f"清空检查点目录: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
    
    if log_dir.exists():
        print(f"清空日志目录: {log_dir}")
        shutil.rmtree(log_dir)


def setup_logging(log_dir: str, log_file: str = 'train.log'):
    """
    设置日志
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_config(config: dict) -> bool:
    """
    验证配置文件
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    logger = logging.getLogger(__name__)
    
    # 检查数据路径
    train_dir = Path(config['data']['train_data_dir'])
    val_dir = Path(config['data']['val_data_dir'])
    
    if not train_dir.exists():
        logger.error(f"训练数据目录不存在: {train_dir}")
        logger.info("请修改配置文件中的 data.train_data_dir")
        return False
    
    if not val_dir.exists():
        logger.warning(f"验证数据目录不存在: {val_dir}")
        logger.warning("将使用训练数据进行验证")
        config['data']['val_data_dir'] = config['data']['train_data_dir']
    
    # 检查 GPU 可用性
    if config['device']['cuda'] and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将使用 CPU 训练")
        config['device']['cuda'] = False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sparse GAN 训练")
    parser.add_argument(
        '--config',
        type=str,
        default='sparse_gan_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default=None,
        help='训练数据目录（覆盖配置文件）'
    )
    parser.add_argument(
        '--val_data_dir',
        type=str,
        default=None,
        help='验证数据目录（覆盖配置文件）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批量大小（覆盖配置文件）'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖配置文件）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（覆盖配置文件）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.train_data_dir is not None:
        config['data']['train_data_dir'] = args.train_data_dir
    if args.val_data_dir is not None:
        config['data']['val_data_dir'] = args.val_data_dir
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir
        config['experiment']['checkpoint_dir'] = str(Path(args.output_dir) / 'checkpoints')
        config['experiment']['log_dir'] = str(Path(args.output_dir) / 'logs')
    
    # 如果不是恢复训练，清空目录
    if args.resume is None:
        clean_directories(config)

    # 设置日志
    setup_logging(config['experiment']['log_dir'])
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Sparse GAN 训练")
    logger.info("="*60)
    
    # 验证配置
    if not validate_config(config):
        logger.error("配置验证失败，退出")
        return
    
    # 打印配置
    logger.info("\n配置:")
    logger.info(f"  实验名称: {config['experiment']['name']}")
    logger.info(f"  输出目录: {config['experiment']['output_dir']}")
    logger.info(f"  训练数据: {config['data']['train_data_dir']}")
    logger.info(f"  验证数据: {config['data']['val_data_dir']}")
    logger.info(f"  体素分辨率: {config['data']['voxel_size']}")
    logger.info(f"  批量大小: {config['training']['batch_size']}")
    logger.info(f"  训练轮数: {config['training']['num_epochs']}")
    logger.info(f"  GAN类型: {config['training']['gan_type']}")
    logger.info(f"  设备: {'CUDA' if config['device']['cuda'] else 'CPU'}")
    
    # 保存配置副本
    config_save_path = Path(config['experiment']['output_dir']) / 'config.yaml'
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"  配置已保存: {config_save_path}")
    
    logger.info("\n" + "="*60)
    
    try:
        # 创建数据模块
        logger.info("初始化数据模块...")
        data_module = SparseGANDataModule(config)
        data_module.setup()
        
        # 创建训练器
        logger.info("初始化训练器...")
        trainer = SparseGANTrainer(config)
        
        # 可视化前两个训练样本
        logger.info("可视化初始训练样本...")
        try:
            train_loader = data_module.get_train_dataloader()
            batch = next(iter(train_loader))
            dense_voxels = batch['dense_voxels'] # [B, C, X, Y, Z]
            num_vis = min(2, dense_voxels.shape[0])
            
            for i in range(num_vis):
                vol = dense_voxels[i].cpu().numpy()[0] # (X, Y, Z) take first channel
                # Project to 2D (Sum projection)
                proj = np.sum(vol, axis=2)
                # Normalize
                proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
                
                trainer.writer.add_image(
                    f"data/sample_{i}",
                    proj,
                    0,
                    dataformats='HW'
                )
            logger.info(f"已将 {num_vis} 个训练样本的可视化结果记录到 TensorBoard")
        except Exception as e:
            logger.warning(f"可视化样本失败: {e}")
        
        # 加载检查点（如果有）
        if args.resume is not None:
            logger.info(f"恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        logger.info("\n" + "="*60)
        logger.info("开始训练")
        logger.info("="*60 + "\n")
        
        trainer.train(data_module)
        
        logger.info("\n" + "="*60)
        logger.info("训练完成！")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
        
    except Exception as e:
        logger.error(f"\n训练过程中出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

