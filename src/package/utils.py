import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from typing import Optional, Dict, Any
import torchvision
from PIL import Image
import numpy as np

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, beta: float):
        self.model = model
        self.beta = beta
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.beta * self.shadow[name]
                    + (1.0 - self.beta) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def save_checkpoint(
    path: Path,
    step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    generator_ema: EMA
):
    """Save model checkpoint."""
    checkpoint = {
        'step': step,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'generator_ema_shadow': generator_ema.shadow
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: str,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    generator_ema: EMA
) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    generator_ema.shadow = checkpoint['generator_ema_shadow']
    
    return checkpoint['step']

def save_image_grid(
    images: torch.Tensor,
    path: Path,
    nrow: int = 8,
    normalize: bool = True
):
    """Save a grid of images."""
    if normalize:
        images = images.mul(0.5).add(0.5).clamp(0, 1)
    
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    Image.fromarray(grid).save(path)

def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: str
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (
        alpha * real_samples + (1 - alpha) * fake_samples
    ).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty