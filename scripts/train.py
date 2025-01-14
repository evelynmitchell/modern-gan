import argparse
from pathlib import Path
import torch
from loguru import logger

from src.package.main import ModelConfig
from src.package.trainer import GANTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Modern GAN")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to training data directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to output directory")
    parser.add_argument("--resume-from", type=str,
                       help="Path to checkpoint to resume from")
    
    # Model config
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    
    # Training config
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=5000)
    
    # Optimizer config
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    
    # Loss config
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        Path(args.output_dir) / "training.log",
        rotation="100 MB"
    )
    
    # Create model config
    config = ModelConfig(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        base_channels=args.base_channels
    )
    
    # Initialize trainer
    trainer = GANTrainer(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        r1_gamma=args.r1_gamma
    )
    
    # Start training
    trainer.train(
        num_steps=args.num_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main()