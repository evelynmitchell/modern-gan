import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
from loguru import logger

from .main import Generator, Discriminator, ModelConfig
from .utils import EMA, compute_gradient_penalty, save_checkpoint, load_checkpoint

class GANTrainer:
    """Modern GAN Trainer implementation."""
    
    def __init__(
        self,
        config: ModelConfig,
        data_dir: str,
        output_dir: str,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        lr: float = 0.002,
        beta1: float = 0.0,
        beta2: float = 0.99,
        r1_gamma: float = 10.0,
        ema_beta: float = 0.999,
        mixed_prob: float = 0.9
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.generator = Generator(config).to(device)
        self.discriminator = Discriminator(config).to(device)
        
        # Setup EMA for generator
        self.generator_ema = EMA(self.generator, beta=ema_beta)
        
        # Setup optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        
        # Training params
        self.r1_gamma = r1_gamma
        self.mixed_prob = mixed_prob
        
        # Setup data
        transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.CenterCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"Initialized trainer with {len(dataset)} images")
    
    def train_step(
        self,
        real_imgs: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """Single training step."""
        batch_size = real_imgs.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real images
        real_imgs.requires_grad = True
        real_pred = self.discriminator(real_imgs)
        d_loss_real = F.softplus(-real_pred).mean()
        
        # R1 gradient penalty
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_imgs,
            create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(batch_size, -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 0.5 * self.r1_gamma * grad_penalty
        
        # Fake images
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        
        # Style mixing regularization
        if np.random.random() < self.mixed_prob:
            z2 = torch.randn(batch_size, self.config.latent_dim).to(self.device)
            mixing_point = np.random.randint(1, self.generator.num_stages)
            fake_imgs = self.generator(z, z2=z2, mixing_point=mixing_point)
        else:
            fake_imgs = self.generator(z)
            
        fake_pred = self.discriminator(fake_imgs.detach())
        d_loss_fake = F.softplus(fake_pred).mean()
        
        d_loss = d_loss_real + d_loss_fake + grad_penalty
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_pred = self.discriminator(fake_imgs)
        g_loss = F.softplus(-fake_pred).mean()
        
        g_loss.backward()
        self.g_optimizer.step()
        
        # Update EMA generator
        self.generator_ema.update()
        
        return {
            "d_loss": d_loss.item(),
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "grad_penalty": grad_penalty.item(),
            "g_loss": g_loss.item()
        }
    
    def train(
        self,
        num_steps: int,
        save_every: int = 1000,
        eval_every: int = 5000,
        resume_from: Optional[str] = None
    ):
        """Training loop."""
        start_step = 0
        if resume_from is not None:
            start_step = load_checkpoint(
                resume_from,
                self.generator,
                self.discriminator,
                self.g_optimizer,
                self.d_optimizer,
                self.generator_ema
            )
            logger.info(f"Resumed from step {start_step}")
        
        dataset_iter = iter(self.dataloader)
        pbar = tqdm(range(start_step, num_steps))
        
        for step in pbar:
            try:
                real_imgs = next(dataset_iter)
                if isinstance(real_imgs, (tuple, list)):
                    real_imgs = real_imgs[0]
            except StopIteration:
                dataset_iter = iter(self.dataloader)
                real_imgs = next(dataset_iter)
                if isinstance(real_imgs, (tuple, list)):
                    real_imgs = real_imgs[0]
            
            real_imgs = real_imgs.to(self.device)
            
            metrics = self.train_step(real_imgs, step)
            
            # Update progress bar
            pbar.set_postfix(
                g_loss=f"{metrics['g_loss']:.4f}",
                d_loss=f"{metrics['d_loss']:.4f}"
            )
            
            # Save checkpoint
            if (step + 1) % save_every == 0:
                save_checkpoint(
                    self.output_dir / f"model_{step+1}.pt",
                    step + 1,
                    self.generator,
                    self.discriminator,
                    self.g_optimizer,
                    self.d_optimizer,
                    self.generator_ema
                )
                logger.info(f"Saved checkpoint at step {step+1}")
            
            # Run evaluation
            if (step + 1) % eval_every == 0:
                self.evaluate(step + 1)
    
    @torch.no_grad()
    def evaluate(self, step: int):
        """Evaluation routine."""
        self.generator_ema.model.eval()
        
        # Generate samples
        num_samples = 64
        z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
        
        # Generate with different truncation values
        for trunc in (0.5, 0.7, 1.0):
            samples = self.generator_ema.model(
                z * trunc
            )
            
            # Save samples grid
            samples_path = self.output_dir / f"samples_step{step}_trunc{trunc}.png"
            save_image_grid(samples, samples_path)
            
        self.generator_ema.model.train()
        logger.info(f"Saved evaluation samples at step {step}")
