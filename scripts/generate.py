import argparse
from pathlib import Path
import torch
from loguru import logger

from src.package.main import ModelConfig
from src.package.inference import GANInference

def main():
    parser = argparse.ArgumentParser(description="Generate images with Modern GAN")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to output directory")
    
    # Model config
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    
    # Generation config
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    
    # Style mixing config
    parser.add_argument("--style-mixing", action="store_true",
                       help="Generate style mixing visualization")
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--num-targets", type=int, default=8)
    
    # Interpolation config
    parser.add_argument("--interpolate", action="store_true",
                       help="Generate latent interpolation")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--end-seed", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        Path(args.output_dir) / "inference.log",
        rotation="100 MB"
    )
    
    # Create model config
    config = ModelConfig(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        base_channels=args.base_channels
    )
    
    # Initialize inference
    inference = GANInference(
        config=config,
        checkpoint_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Generate samples
    if not args.style_mixing and not args.interpolate:
        inference.generate_images(
            num_samples=args.num_samples,
            truncation=args.truncation,
            seed=args.seed,
            output_dir=args.output_dir
        )
    
    # Generate style mixing visualization
    if args.style_mixing:
        inference.style_mixing(
            num_samples=args.num_targets,
            source_seed=args.source_seed,
            truncation=args.truncation,
            output_dir=args.output_dir
        )
    
    # Generate interpolation
    if args.interpolate:
        inference.interpolate(
            start_seed=args.start_seed,
            end_seed=args.end_seed,
            num_steps=args.num_steps,
            truncation=args.truncation,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()