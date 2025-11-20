"""
Sparse GAN 训练器

实现 WGAN-GP 训练循环。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import MinkowskiEngine as ME
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import yaml
import numpy as np

from sparse_gan_models import SparseGenerator, SparseDiscriminator, compute_gradient_penalty
from sparse_gan_dataset import SparseGANDataModule, create_sparse_tensor_from_batch, sparse_to_dense

logger = logging.getLogger(__name__)


class SparseGANTrainer:
    """
    Sparse GAN 训练器
    
    支持 WGAN-GP、Vanilla GAN、LSGAN 等多种 GAN 训练方式。
    """
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = self._setup_device()
        
        # 创建输出目录
        self.output_dir = Path(config['experiment']['output_dir'])
        self.checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
        self.log_dir = Path(config['experiment']['log_dir'])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 初始化模型
        self.generator = self._create_generator().to(self.device)
        self.discriminator = self._create_discriminator().to(self.device)
        
        # 初始化优化器
        self.g_optimizer = self._create_optimizer(
            self.generator,
            config['training']['generator_optimizer']
        )
        self.d_optimizer = self._create_optimizer(
            self.discriminator,
            config['training']['discriminator_optimizer']
        )
        
        # 初始化学习率调度器
        self.g_scheduler = self._create_scheduler(self.g_optimizer)
        self.d_scheduler = self._create_scheduler(self.d_optimizer)
        
        # 训练参数
        self.gan_type = config['training']['gan_type']
        self.n_critic = config['training']['n_critic']
        self.gp_weight = config['training']['gradient_penalty_weight']
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        logger.info(f"训练器初始化完成 (设备: {self.device})")
        logger.info(f"GAN类型: {self.gan_type}")
        logger.info(f"生成器参数量: {sum(p.numel() for p in self.generator.parameters()):,}")
        logger.info(f"判别器参数量: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if self.config['device']['cuda'] and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("使用 CPU")
        return device
    
    def _create_generator(self) -> nn.Module:
        """创建生成器"""
        gen_config = self.config['generator']
        return SparseGenerator(
            latent_dim=gen_config['latent_dim'],
            channels=gen_config['channels'],
            output_channels=gen_config['output_channels'],
            initial_tensor_stride=gen_config['initial_tensor_stride'],
            resolution=self.config['data']['voxel_size'],
            kernel_size=gen_config['kernel_size'],
            activation=gen_config['activation'],
            use_batch_norm=gen_config['use_batch_norm']
        )
    
    def _create_discriminator(self) -> nn.Module:
        """创建判别器"""
        disc_config = self.config['discriminator']
        gen_config = self.config['generator']
        return SparseDiscriminator(
            input_channels=gen_config['output_channels'],
            channels=disc_config['channels'],
            kernel_size=disc_config['kernel_size'],
            activation=disc_config['activation'],
            use_batch_norm=disc_config['use_batch_norm'],
            use_spectral_norm=disc_config['use_spectral_norm']
        )
    
    def _create_optimizer(self, model: nn.Module, opt_config: Dict) -> optim.Optimizer:
        """创建优化器"""
        if opt_config['type'].lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'].lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer):
        """创建学习率调度器"""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'] == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_config['gamma']
            )
        elif sched_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif sched_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def train_discriminator_step(
        self,
        real_sparse: ME.SparseTensor,
        fake_sparse: ME.SparseTensor
    ) -> Dict[str, float]:
        """
        训练判别器一步
        
        Args:
            real_sparse: 真实稀疏张量
            fake_sparse: 生成的稀疏张量
            
        Returns:
            损失字典
        """
        self.d_optimizer.zero_grad()
        
        # 判别器预测
        d_real = self.discriminator(real_sparse)
        d_fake = self.discriminator(fake_sparse.detach())
        
        if self.gan_type == 'wgan-gp':
            # WGAN-GP 损失
            d_loss = d_fake.mean() - d_real.mean()
            
            # 计算梯度惩罚
            gp = compute_gradient_penalty(
                self.discriminator,
                real_sparse,
                fake_sparse,
                self.device
            )
            d_loss += self.gp_weight * gp
            
            # Wasserstein distance
            wd = -d_loss.item() + self.gp_weight * gp.item()
            
            losses = {
                'd_loss': d_loss.item(),
                'd_real': d_real.mean().item(),
                'd_fake': d_fake.mean().item(),
                'gradient_penalty': gp.item(),
                'wasserstein_distance': wd
            }
        
        elif self.gan_type == 'vanilla':
            # Vanilla GAN 损失（BCE）
            criterion = nn.BCEWithLogitsLoss()
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)
            
            d_loss_real = criterion(d_real, real_labels)
            d_loss_fake = criterion(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            
            losses = {
                'd_loss': d_loss.item(),
                'd_loss_real': d_loss_real.item(),
                'd_loss_fake': d_loss_fake.item()
            }
        
        elif self.gan_type == 'lsgan':
            # Least Squares GAN 损失
            d_loss_real = 0.5 * ((d_real - 1) ** 2).mean()
            d_loss_fake = 0.5 * (d_fake ** 2).mean()
            d_loss = d_loss_real + d_loss_fake
            
            losses = {
                'd_loss': d_loss.item(),
                'd_loss_real': d_loss_real.item(),
                'd_loss_fake': d_loss_fake.item()
            }
        
        else:
            raise ValueError(f"不支持的 GAN 类型: {self.gan_type}")
        
        # 反向传播
        d_loss.backward()
        self.d_optimizer.step()
        
        return losses
    
    def train_generator_step(self, batch_size: int) -> Dict[str, float]:
        """
        训练生成器一步
        
        Args:
            batch_size: 批量大小
            
        Returns:
            损失字典
        """
        self.g_optimizer.zero_grad()
        
        # 采样潜在向量
        z = torch.randn(
            batch_size,
            self.config['generator']['latent_dim'],
            device=self.device
        )
        
        # 生成假样本
        fake_sparse = self.generator(z)
        
        # 判别器预测
        d_fake = self.discriminator(fake_sparse)
        
        if self.gan_type == 'wgan-gp':
            # WGAN 损失：最大化判别器对假样本的分数
            g_loss = -d_fake.mean()
            
            losses = {
                'g_loss': g_loss.item(),
                'd_fake_for_g': d_fake.mean().item()
            }
        
        elif self.gan_type == 'vanilla':
            # Vanilla GAN 损失
            criterion = nn.BCEWithLogitsLoss()
            real_labels = torch.ones_like(d_fake)
            g_loss = criterion(d_fake, real_labels)
            
            losses = {
                'g_loss': g_loss.item()
            }
        
        elif self.gan_type == 'lsgan':
            # LSGAN 损失
            g_loss = 0.5 * ((d_fake - 1) ** 2).mean()
            
            losses = {
                'g_loss': g_loss.item()
            }
        
        else:
            raise ValueError(f"不支持的 GAN 类型: {self.gan_type}")
        
        # 反向传播
        g_loss.backward()
        self.g_optimizer.step()
        
        return losses
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            平均损失字典
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'd_loss': 0.0,
            'g_loss': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch_size = batch['dense_voxels'].size(0)
            
            # 创建真实稀疏张量
            real_sparse = create_sparse_tensor_from_batch(batch, self.device)
            
            # ===== 训练判别器 =====
            # 生成假样本
            z = torch.randn(
                batch_size,
                self.config['generator']['latent_dim'],
                device=self.device
            )
            fake_sparse = self.generator(z)
            
            # 训练判别器
            d_losses = self.train_discriminator_step(real_sparse, fake_sparse)
            
            # ===== 训练生成器 =====
            # 每 n_critic 步训练一次生成器
            if (batch_idx + 1) % self.n_critic == 0:
                g_losses = self.train_generator_step(batch_size)
            else:
                g_losses = {'g_loss': 0.0}
            
            # 累积损失
            epoch_losses['d_loss'] += d_losses['d_loss']
            epoch_losses['g_loss'] += g_losses['g_loss']
            
            # 更新进度条
            pbar.set_postfix({
                'D': f"{d_losses['d_loss']:.4f}",
                'G': f"{g_losses['g_loss']:.4f}"
            })
            
            # 记录到 TensorBoard
            if self.global_step % self.config['training']['log_interval'] == 0:
                for key, value in d_losses.items():
                    self.writer.add_scalar(f'train/discriminator/{key}', value, self.global_step)
                for key, value in g_losses.items():
                    self.writer.add_scalar(f'train/generator/{key}', value, self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前 epoch
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"加载检查点: {checkpoint_path} (epoch {self.current_epoch})")
    
    def train(self, data_module: SparseGANDataModule):
        """
        完整训练流程
        
        Args:
            data_module: 数据模块
        """
        logger.info("开始训练 Sparse GAN...")
        
        train_loader = data_module.get_train_dataloader()
        val_loader = data_module.get_val_dataloader()
        num_epochs = self.config['training']['num_epochs']
        
        # 获取固定验证批次用于可视化
        fixed_val_batch = None
        if val_loader:
            try:
                fixed_val_batch = next(iter(val_loader))
                logger.info("已获取固定验证批次用于可视化对比")
            except StopIteration:
                logger.warning("验证集为空，可视化将不包含真实样本对比")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # 训练一个 epoch
            epoch_losses = self.train_epoch(train_loader)
            
            logger.info(f"Epoch {epoch} 完成:")
            logger.info(f"  判别器损失: {epoch_losses['d_loss']:.4f}")
            logger.info(f"  生成器损失: {epoch_losses['g_loss']:.4f}")
            
            # 学习率调度
            if self.g_scheduler is not None:
                self.g_scheduler.step()
            if self.d_scheduler is not None:
                self.d_scheduler.step()
            
            # 记录学习率
            self.writer.add_scalar('train/g_lr', self.g_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('train/d_lr', self.d_optimizer.param_groups[0]['lr'], epoch)
            
            # 保存检查点
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # 生成样本
            if (epoch + 1) % self.config['training']['sample_interval'] == 0:
                self.generate_samples(epoch, fixed_val_batch)
        
        logger.info("\n训练完成！")
        self.writer.close()
    
    @torch.no_grad()
    def generate_samples(self, epoch: int, val_batch: Optional[Dict] = None):
        """
        生成样本并可视化
        
        Args:
            epoch: 当前 epoch
            val_batch: 验证批次（包含真实样本用于对比）
        """
        self.generator.eval()
        
        num_samples = self.config['training']['num_samples']
        
        # 如果有真实样本，调整生成的数量以匹配
        if val_batch is not None:
            real_dense = val_batch['dense_voxels']
            num_samples = min(num_samples, real_dense.shape[0])
        
        # 采样潜在向量
        z = torch.randn(
            num_samples,
            self.config['generator']['latent_dim'],
            device=self.device
        )
        
        # 生成样本
        fake_sparse = self.generator(z)
        
        # 可视化并记录到 TensorBoard
        for i in range(num_samples):
            # 获取生成样本 (C, X, Y, Z)
            synthetic_tensor = sparse_to_dense(
                fake_sparse,
                resolution=self.config['data']['voxel_size'],
                batch_idx=i
            )
            synthetic_vol = synthetic_tensor.cpu().numpy()[0]  # (X, Y, Z)
            
            # 投影 (X, Y)
            synthetic_proj = np.sum(synthetic_vol, axis=2)
            
            # 归一化生成样本
            synthetic_proj = (synthetic_proj - synthetic_proj.min()) / (synthetic_proj.max() - synthetic_proj.min() + 1e-8)
            
            if val_batch is not None:
                # 获取真实样本 (C, X, Y, Z)
                real_tensor = val_batch['dense_voxels'][i]
                real_vol = real_tensor.cpu().numpy()[0]  # (X, Y, Z)
                
                # 投影
                real_proj = np.sum(real_vol, axis=2)
                
                # 归一化真实样本
                real_proj = (real_proj - real_proj.min()) / (real_proj.max() - real_proj.min() + 1e-8)
                
                # 拼接
                combined = np.hstack([real_proj, synthetic_proj])
            else:
                combined = synthetic_proj
            
            # 添加到 TensorBoard
            self.writer.add_image(
                f"vis/sample_{i}",
                combined,
                epoch,
                dataformats='HW'
            )
        
        logger.info(f"已保存 {num_samples} 个样本的可视化结果到 TensorBoard")
        
        self.generator.train()


if __name__ == "__main__":
    """训练脚本"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Sparse GAN 训练")
    parser.add_argument('--config', type=str, default='sparse_gan_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器
    trainer = SparseGANTrainer(config)
    
    # 加载检查点（如果有）
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # 创建数据模块
    data_module = SparseGANDataModule(config)
    data_module.setup()
    
    # 开始训练
    trainer.train(data_module)

