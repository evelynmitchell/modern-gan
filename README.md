

# Modern GANs

An implementation of The GAN is dead; long live the GAN! A Modern GAN Baseline (https://arxiv.org/abs/2501.05441) in PyTorch


## Installation

I've given instructions for either uv or pip. You can use either of them.i

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

### Code Quality 🧹


This project uses uv and pre-commit hooks.

Once on project set up run:
```
python -m venv /path/to/new/virtual/environment
```
where /path/to/new/virtual/environment is the path to the new virtual environment you want to create. I suggest /project/.venv as a good idea.

Then activate the virtual environment by running:
```
source .venv/bin/activate
```

When you are done working in this venv you can deactivate it by running:
``` 
source deactivate
```


We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`

### Tests 🧪

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.

### Publish on PyPi 🚀

**Important**: Before publishing, edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.

```
poetry build
poetry publish
```

### CI/CD 🤖

We use [GitHub actions](https://github.com/features/actions) to automatically run tests and check code quality when a new PR is done on `main`.

On any pull request, we will check the code quality and tests.

When a new release is created, we will try to push the new code to PyPi. We use [`twine`](https://twine.readthedocs.io/en/stable/) to make our life easier. 

The **correct steps** to create a new realease are the following:
- edit `__version__` in [src/__init__](/src/__init__.py) to match the wanted new version.
- create a new [`tag`](https://git-scm.com/docs/git-tag) with the release name, e.g. `git tag v0.0.1 && git push origin v0.0.1` or from the GitHub UI.
- create a new release from GitHub UI

The CI will run when you create the new release.

# Docs
We use MK docs. This repo comes with the zeta docs. All the docs configurations are already here along with the readthedocs configs.



# License
MIT
