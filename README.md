

# Modern GANs

An implementation of The GAN is dead; long live the GAN! A Modern GAN Baseline (https://arxiv.org/abs/2501.05441) in PyTorch


## Installation

I've given instructions for either uv or pip. You can use either of them.

The colab notebook is the easiest: modern-gan/modern_gan.ipynb

### uv Installation

Checkout the code and run from source: 

```bash
git clone https://github.com/evelynmitchell/modern-gan.git
cd modern-gan
uv run --python 3.12 --with torch --with loguru modern-gan/src/package/main.py --system
```

### pip Installation

Create a virtual environment for python:

```bash
python -m venv .venv
source .venv/bin/activate
```

Checkout the code and install from source: 

```bash
git clone https://github.com/evelynmitchell/modern-gan.git
cd modern-gan
pip install . --system
```

## Generate Image Samples

The StyeGAN2 model (baseline) is trained on the FFHQ dataset (https://huggingface.co/datasets/student/FFHQ). There are other versions of StyleGAN at https://nvlabs.github.io/stylegan2/versions.html

The code for generating samples in the StyleGAN2 paper is proprietary to NVIDIA. We can generate samples using another method. If you want to replicate the original paper, you will need to use the original code.


# Usage
```bash
git clone https://github.com/evelynmitchell/modern-gan.git
cd modern-gan
uv run --python 3.12  modern-gan/src/package/main.py --system
```

```
I've created a complete implementation that includes:

1. Training Pipeline (`trainer.py`):
   - Full training loop with gradient penalty and style mixing
   - EMA model averaging for better quality
   - Checkpointing and logging
   - Evaluation during training

2. Inference Pipeline (`inference.py`):
   - Image generation with truncation trick
   - Style mixing visualization
   - Latent space interpolation

3. Utility Functions (`utils.py`):
   - EMA implementation
   - Checkpoint saving/loading
   - Image saving utilities
   - Gradient penalty computation

4. Scripts:
   - `train.py` for training the model
   - `generate.py` for inference and visualization

The key improvements over the original implementation include:

1. Training Features:
   - R1 gradient penalty for stable training
   - Style mixing regularization
   - EMA model averaging
   - Proper data loading and augmentation

2. Advanced Generation:
   - Truncation trick for quality vs diversity control
   - Style mixing for controlled attribute editing
   - Latent space interpolation

To use this implementation:

1. Training:
```bash
python scripts/train.py \
    --data-dir /path/to/dataset \
    --output-dir /path/to/output \
    --img-size 256 \
    --batch-size 32 \
    --num-steps 100000
```

2. Generation:
```bash
python scripts/generate.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir /path/to/output \
    --num-samples 64 \
    --truncation 0.7
```

3. Style Mixing:
```bash
python scripts/generate.py \
    --checkpoint /path/to/checkpoint.pt \
    --output-dir /path/to/output \
    --style-mixing \
    --num-targets 8
```

The implementation follows modern GAN training practices and includes features from recent papers like:
- Gradient penalty for stability
- Style mixing for better disentanglement
- EMA for higher quality results
- Truncation trick for controlling the quality-diversity tradeoff

Would you like me to explain any specific part in more detail or help you get started with training?
```

### CI/CD 🤖

We use [GitHub actions](https://github.com/features/actions) to automatically run tests and check code quality when a new PR is done on `main`.

On any pull request, we will check the code quality and tests.

When a new release is created, we will try to push the new code to PyPi. We use [`twine`](https://twine.readthedocs.io/en/stable/) to make our life easier. 

The **correct steps** to create a new realease are the following:
- edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.
- create a new [`tag`](https://git-scm.com/docs/git-tag) with the release name, e.g. `git tag v0.0.1 && git push origin v0.0.1` or from the GitHub UI.
- create a new release from GitHub UI

The CI will run when you commit a change.


# License
MIT
