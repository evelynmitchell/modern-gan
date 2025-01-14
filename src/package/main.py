import torch
from torch import nn
from typing import List
from loguru import logger
from dataclasses import dataclass
# noqa: C0303
@dataclass
class ModelConfig:
    """Configuration for the GAN model.
    
    Attributes:
        latent_dim: Dimension of the input noise vector
        img_channels: Number of channels in the generated image
        base_channels: Base number of channels (width) in the network
        img_size: Size of the generated image (assumes square)
        group_size: Size of groups for grouped convolutions (default: 16)
    """
    latent_dim: int = 512
    img_channels: int = 3
    base_channels: int = 32
    img_size: int = 256
    group_size: int = 16

class ResidualBlock(nn.Module):
    """Modern residual block with grouped convolutions and inverted bottleneck.
    
    Args:
        in_channels: Number of input channels
        compression_ratio: Ratio for bottleneck compression
        group_size: Size of groups for grouped convolutions
    """
    def __init__(
        self,
        in_channels: int,
        compression_ratio: float = 2.0,
        group_size: int = 16
    ):
        super().__init__()
        bottleneck_channels = int(in_channels * 1.5)  # Inverted bottleneck
        groups = max(1, bottleneck_channels // group_size)
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=True)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            groups=groups,
            bias=True
        )
        self.act2 = nn.LeakyReLU(0.2)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, 1, bias=False)
        
        # Initialize using fix-up
        nn.init.zeros_(self.conv3.weight)
        scale = 0.25  # Will be adjusted based on depth later
        nn.init.normal_(self.conv1.weight, std=scale)
        nn.init.normal_(self.conv2.weight, std=scale)
        
        logger.debug(f"Created ResBlock: {in_channels} -> {bottleneck_channels} -> {in_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        
        return out + identity

class TransitionBlock(nn.Module):
    """Transition block for changing spatial dimensions and channel count.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Factor to scale spatial dimensions (2 for up, 0.5 for down)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.needs_conv = in_channels != out_channels
        
        if self.needs_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 1)
            
        logger.debug(
            f"Created TransitionBlock: {in_channels} -> {out_channels}, "
            f"scale: {scale_factor}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Separate resampling layer
        if self.scale_factor > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False
            )
        elif self.scale_factor < 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False
            )
            
        if self.needs_conv:
            x = self.conv(x)
            
        return x

class Generator(nn.Module):
    """Modern Generator network.
    
    Args:
        config: Model configuration
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Calculate number of resolution stages
        self.num_stages = int(torch.log2(torch.tensor(config.img_size))) - 2
        channels_list = self._get_channels_list()
        
        # 4x4 learnable basis layer
        self.basis = nn.Parameter(
            torch.randn(1, channels_list[0], 4, 4)
        )
        self.basis_mod = nn.Linear(config.latent_dim, channels_list[0])
        
        # Build resolution stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = self._make_stage(
                channels_list[i],
                channels_list[i + 1],
                is_last=(i == self.num_stages - 1)
            )
            self.stages.append(stage)
            
        logger.info(f"Created Generator with {self.num_stages} stages")

    def _get_channels_list(self) -> List[int]:
        """Calculate channels for each resolution stage."""
        base = self.config.base_channels
        return [base * (2 ** (self.num_stages - i)) for i in range(self.num_stages + 1)]

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        is_last: bool
    ) -> nn.ModuleList:
        """Create a resolution stage."""
        blocks = nn.ModuleList([
            TransitionBlock(in_channels, out_channels, scale_factor=2),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        ])
        
        if is_last:
            blocks.append(
                nn.Conv2d(out_channels, self.config.img_channels, 1)
            )
            blocks.append(nn.Tanh())
            
        return blocks

    def forward(
        self,
        z: torch.Tensor,
        z2: Optional[torch.Tensor] = None,
        mixing_point: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            z: Input noise tensor of shape [batch_size, latent_dim]
            z2: Optional second noise tensor for style mixing
            mixing_point: If provided with z2, switch to z2 latents after this stage
            
        Returns:
            Generated images of shape [batch_size, img_channels, img_size, img_size]
        """
        batch_size = z.shape[0]
        
        # Modulate basis with input
        mod = self.basis_mod(z).view(batch_size, -1, 1, 1)
        x = self.basis.repeat(batch_size, 1, 1, 1) * mod
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            # Switch to second latent if mixing
            if z2 is not None and mixing_point is not None and i == mixing_point:
                z = z2
            
            for block in stage:
                x = block(x)
                logger.trace(f"Shape after block: {x.shape}")
                
        return x

class Discriminator(nn.Module):
    """Modern Discriminator network.
    
    Args:
        config: Model configuration
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.num_stages = int(torch.log2(torch.tensor(config.img_size))) - 2
        channels_list = self._get_channels_list()
        
        # Build resolution stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = self._make_stage(
                channels_list[i],
                channels_list[i + 1],
                is_first=(i == 0)
            )
            self.stages.append(stage)
            
        # Classifier head - properly reshape before linear layer
        self.head = nn.Sequential(
            nn.Conv2d(
                channels_list[-1],
                channels_list[-1],
                4,
                groups=channels_list[-1]
            ),
            nn.Flatten(),  # Flatten the spatial dimensions
            nn.Linear(channels_list[-1], 1)
        )
        
        logger.info(f"Created Discriminator with {self.num_stages} stages")

    def _get_channels_list(self) -> List[int]:
        """Calculate channels for each resolution stage."""
        base = self.config.base_channels
        return [base * (2 ** i) for i in range(self.num_stages + 1)]

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        is_first: bool
    ) -> nn.ModuleList:
        """Create a resolution stage."""
        blocks = nn.ModuleList()
        
        if is_first:
            blocks.append(
                nn.Conv2d(self.config.img_channels, in_channels, 1)
            )
            
        blocks.extend([
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            TransitionBlock(in_channels, out_channels, scale_factor=0.5)
        ])
            
        return blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape [batch_size, img_channels, img_size, img_size]
            
        Returns:
            Discrimination scores of shape [batch_size, 1]
        """
        # Process through stages
        for stage in self.stages:
            for block in stage:
                x = block(x)
                logger.trace(f"Shape after block: {x.shape}")
                
        # Global depthwise conv and classification
        x = self.head(x).squeeze(-1).squeeze(-1)
        return x

# Example usage
if __name__ == "__main__":
    logger.add("model.log", level="DEBUG")
    
    config = ModelConfig(
        latent_dim=512,
        img_channels=3,
        base_channels=32,
        img_size=256
    )
    
    generator = Generator(config)
    discriminator = Discriminator(config)
    
    # Test forward pass
    z = torch.randn(4, config.latent_dim)
    fake_images = generator(z)
    scores = discriminator(fake_images)
    
    logger.info(f"Generated image shape: {fake_images.shape}")
    logger.info(f"Discrimination scores shape: {scores.shape}")
