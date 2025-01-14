import torch
from torch import nn
from typing import Optional, List
from pathlib import Path
from loguru import logger

from .main import Generator, ModelConfig
from .utils import EMA, save_image_grid

class GANInference:
    """Inference class for the trained GAN model."""
    
    def __init__(
        self,
        config: ModelConfig,
        checkpoint_path: str,
        device: str = "cuda"
    ):
        self.config = config
        self.device = device
        
        # Initialize model
        self.generator = Generator(config).to(device)
        self.generator.eval()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        # Setup EMA if available
        if 'generator_ema_shadow' in checkpoint:
            self.generator_ema = EMA(self.generator, beta=0.999)
            self.generator_ema.shadow = checkpoint['generator_ema_shadow']
            self.generator_ema.apply_shadow()
            logger.info("Loaded EMA weights")
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    @torch.no_grad()
    def generate_images(
        self,
        num_samples: int = 1,
        truncation: float = 0.7,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Generate images with the model.
        
        Args:
            num_samples: Number of images to generate
            truncation: Truncation trick value (0.7 is a good default)
            seed: Random seed for reproducibility
            output_dir: If provided, save images to this directory
            
        Returns:
            Tensor of generated images [N, C, H, W]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
        z = truncation * z
        
        images = self.generator(z)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img in enumerate(images):
                save_image_grid(
                    img.unsqueeze(0),
                    output_dir / f"sample_{i:04d}.png"
                )
        
        return images
    
    @torch.no_grad()
    def style_mixing(
        self,
        num_samples: int = 1,
        source_seed: int = 0,
        target_seeds: Optional[List[int]] = None,
        mixing_layers: Optional[List[int]] = None,
        truncation: float = 0.7,
        output_dir: Optional[str] = None
    ) -> List[torch.Tensor]:
        """Generate style mixing visualization.
        
        Args:
            num_samples: Number of target styles to generate
            source_seed: Seed for source style
            target_seeds: Seeds for target styles (if None, random seeds used)
            mixing_layers: Which layers to mix (if None, try all combinations)
            truncation: Truncation trick value
            output_dir: If provided, save visualization to this directory
            
        Returns:
            List of generated image tensors
        """
        torch.manual_seed(source_seed)
        z_source = torch.randn(1, self.config.latent_dim).to(self.device)
        
        if target_seeds is None:
            target_seeds = list(range(num_samples))
        
        z_targets = []
        for seed in target_seeds:
            torch.manual_seed(seed)
            z_targets.append(
                torch.randn(1, self.config.latent_dim).to(self.device)
            )
        
        if mixing_layers is None:
            mixing_layers = list(range(self.generator.num_stages))
        
        results = []
        for z_target in z_targets:
            row_results = [
                self.generator(z_source * truncation)  # Source image
            ]
            
            for mix_layer in mixing_layers:
                mixed = self.generator(
                    z_source * truncation,
                    z2=z_target * truncation,
                    mixing_point=mix_layer
                )
                row_results.append(mixed)
            
            row_results.append(
                self.generator(z_target * truncation)  # Target image
            )
            results.extend(row_results)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create style mixing visualization grid
            grid = torch.cat(results, dim=0)
            save_image_grid(
                grid,
                output_dir / "style_mixing.png",
                nrow=len(mixing_layers) + 2
            )
        
        return results
    
    @torch.no_grad()
    def interpolate(
        self,
        start_seed: int,
        end_seed: int,
        num_steps: int = 10,
        truncation: float = 0.7,
        output_dir: Optional[str] = None
    ) -> torch.Tensor:
        """Generate latent space interpolation.
        
        Args:
            start_seed: Seed for start point
            end_seed: Seed for end point
            num_steps: Number of interpolation steps
            truncation: Truncation trick value
            output_dir: If provided, save interpolation to this directory
            
        Returns:
            Tensor of interpolated images [num_steps, C, H, W]
        """
        # Generate endpoints
        torch.manual_seed(start_seed)
        z_start = torch.randn(1, self.config.latent_dim).to(self.device)
        
        torch.manual_seed(end_seed)
        z_end = torch.randn(1, self.config.latent_dim).to(self.device)
        
        # Generate interpolation points
        alphas = torch.linspace(0, 1, num_steps).to(self.device)
        z_interp = torch.zeros(num_steps, self.config.latent_dim).to(self.device)
        
        for i, alpha in enumerate(alphas):
            z_interp[i] = (1 - alpha) * z_start + alpha * z_end
        
        # Generate images
        images = self.generator(z_interp * truncation)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_image_grid(
                images,
                output_dir / "interpolation.png",
                nrow=num_steps
            )
        
        return images