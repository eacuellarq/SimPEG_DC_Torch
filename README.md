# SimPEG DC Resistivity with PyTorch Backend

A PyTorch implementation of SimPEG's DC resistivity simulations with automatic differentiation.

## Overview

This repository provides PyTorch-enabled versions of SimPEG's core modules, specifically optimized for DC resistivity simulations. The implementation leverages PyTorch's automatic differentiation capabilities.

## Features

- PyTorch backend for SimPEG DC resistivity simulations in 2.5D and 3D.
- GPU support
- Automatic differentiation for gradient-based inversions
- Custom solvers optimized for sparse linear systems
- Selective replacement of discretize modules while preserving compiled extensions
- Compatible with existing SimPEG workflows

## Requirements

- Python 3.10
- CUDA-capable GPU (optional)
- Anaconda or Miniconda

## Installation

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate simpeg-pytorch
```

### Step 2: Install SimPEG

```bash
pip install simpeg==0.18.1
```

### Step 3: Install PyTorch

Choose the appropriate PyTorch installation for your system:

**For CUDA 12.4 (GPU acceleration):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

**For other CUDA versions, see:** https://pytorch.org/get-started/locally/

### Step 4: Install pydiso

```bash
conda install -c conda-forge pydiso
```

### Step 5: Install PyTorch Modifications

```bash
python install.py
```

### Verification

Test the installation:

```python
import SimPEG
from SimPEG.config import SimpegConfig

# Check if PyTorch backend is active
cfg = SimpegConfig()
print(f"PyTorch backend active: {cfg.torch_is_active}")

# Test basic functionality
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick Start

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate simpeg-pytorch

# 2. Install SimPEG
pip install simpeg==0.18.1

# 3. Install PyTorch (choose GPU or CPU version)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install pydiso
conda install -c conda-forge pydiso

# 5. Install PyTorch modifications
python install.py
```

## Architecture

The implementation modifies the following SimPEG components:

- **SimPEG Core**: Complete replacement with PyTorch tensor operations
- **Discretize**: Selective replacement preserving compiled extensions
- **Custom Solvers**: Pardiso and SuperLU adaptations (only CPU) but also you can create a custom solver with GPU if you want and use with SimPEG.   
- **Utilities**: Enhanced simulation utilities in `utils/` module

## Examples

See the `examples/` directory for complete working examples:

- `fwd_dcr_plane_2d.ipynb`: DC resistivity forward modeling in 2D plane geometry
- `fwd_dcr_topo_2d.ipynb`: DC resistivity forward modeling with 2D topography
- `fwd_dcr_topo_3d.ipynb`: DC resistivity forward modeling with 3D topography

## Performance

Preliminary benchmarks show:

  - Faster and better gradients than aproximation of SimPEG vanilla for Direct Current simulations in 2.5D and 3D.
## Limitations

- Currently supports DC resistivity simulations only
- GPU is compatible but don't have a good solver for that
- Some advanced SimPEG features may not be fully compatible because this development was in SimPEG 0.18.1

## Contributing

This is research extension under active development. Please report issues or contribute improvements via GitHub.

## Citation

If you use this extension in your research, please cite:

```bibtex
@extension{simpeg_dc_pytorch,
  title = {SimPEG DC Resistivity with PyTorch Backend},
  author = {Edwin Cuellar},
  year = {2025},
  url = {https://github.com/eacuellarq/simpeg-dc-pytorch},
  note = {Research extension for PyTorch geophysical simulations}
}
```
## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work builds upon the SimPEG framework:

- Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85, 142-154.

## Support
For questions and support:

- Open an issue on GitHub
- Contact: eaqm1228@hotmail.com

