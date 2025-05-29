# Forge Dream Extension - Complete Installation Guide

## Overview

The Forge Dream Extension provides a dual-panel UI for Stable Diffusion Forge with support for HiDream FP8/GGUF models, face swapping capabilities, and CUDA 12.8 acceleration. This guide covers the complete installation process including numpy compatibility fixes.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8 - 3.11 (3.12 not fully supported yet)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and dependencies

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: GTX 1060 6GB or better
- **CUDA**: 12.8 (latest supported)
- **VRAM**: 6GB minimum, 12GB+ recommended for large models
- **Compute Capability**: 6.1 or higher

### Supported Hardware
- ✅ RTX 5070 Ti, RTX 4090, RTX 4080, RTX 4070
- ✅ RTX 3090, RTX 3080, RTX 3070
- ✅ RTX 2080 Ti, RTX 2070 Super
- ✅ GTX 1080 Ti, GTX 1070 (limited performance)

## Quick Installation

### Method 1: Automatic Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/Valorking6/forge-dream-extension.git
cd forge-dream-extension

# Run the enhanced installer
python install.py
```

The installer will automatically:
- ✅ Fix numpy compatibility issues
- ✅ Install PyTorch with CUDA 12.8
- ✅ Setup all dependencies
- ✅ Configure the environment
- ✅ Verify the installation

### Method 2: Manual Installation

If you prefer manual control or the automatic installer fails:

```bash
# Step 1: Fix numpy compatibility
python fix_numpy_compatibility.py

# Step 2: Install PyTorch with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 3: Install requirements
pip install -r requirements.txt

# Step 4: Install optional components
pip install xformers>=0.0.20
pip install insightface>=0.7.3 onnxruntime-gpu>=1.15.0
```

## Detailed Installation Steps

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create new environment
conda create -n forge_dream python=3.10
conda activate forge_dream

# Install base packages
conda install numpy=1.24.4 scipy=1.10.1
```

#### Option B: Using venv
```bash
# Create virtual environment
python -m venv forge_dream_env

# Activate environment
source forge_dream_env/bin/activate  # Linux/macOS
# or
forge_dream_env\Scripts\activate     # Windows
```

### Step 2: Core Dependencies

```bash
# Install numpy with compatibility fixes
pip install --no-cache-dir --force-reinstall "numpy>=1.24.0,<2.0.0"
pip install --no-cache-dir --force-reinstall "scipy>=1.10.0,<1.12.0"
```

### Step 3: PyTorch Installation

#### For CUDA 12.8 (Recommended)
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### For CUDA 11.8 (Fallback)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Extension Dependencies

```bash
# Core ML libraries
pip install transformers>=4.30.0
pip install diffusers>=0.18.0
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0

# Computer vision
pip install opencv-python>=4.8.0
pip install Pillow>=9.5.0

# Memory optimization
pip install xformers>=0.0.20

# Face processing (optional)
pip install insightface>=0.7.3
pip install onnxruntime-gpu>=1.15.0

# Quantized models (optional)
pip install llama-cpp-python>=0.1.78
```

### Step 5: Verification

```bash
# Test the installation
python -c "
import torch
import numpy as np
import scipy
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
"
```

## Configuration

### Basic Configuration

Create or edit `config.json`:

```json
{
  "model_path": "./models",
  "output_path": "./outputs",
  "cuda_enabled": true,
  "memory_optimization": true,
  "face_swap_enabled": true,
  "max_memory_gb": 12,
  "precision": "fp16",
  "safety_checker": true
}
```

### Advanced Configuration

```json
{
  "model_path": "./models",
  "output_path": "./outputs",
  "temp_path": "./temp",
  "cuda_enabled": true,
  "device": "auto",
  "memory_optimization": true,
  "attention_slicing": true,
  "cpu_offload": false,
  "face_swap_enabled": true,
  "face_swap_model": "buffalo_l",
  "max_memory_gb": 12,
  "precision": "fp16",
  "compile_model": false,
  "safety_checker": true,
  "nsfw_filter": true,
  "watermark": false,
  "logging_level": "INFO"
}
```

## Usage

### Basic Usage

```python
from forge_dream_fixed import ForgeDreamExtension

# Initialize the extension
extension = ForgeDreamExtension()

# Load a model
extension.load_model("path/to/your/model.safetensors")

# Generate an image
result = extension.generate(
    prompt="a beautiful landscape",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
    steps=20,
    guidance_scale=7.5
)

# Save the result
result.save("output.png")
```

### Face Swap Usage

```python
# Enable face swap
extension.enable_face_swap()

# Perform face swap
swapped_image = extension.face_swap(
    source_image="source.jpg",
    target_image="target.jpg"
)

swapped_image.save("swapped_result.jpg")
```

### Batch Processing

```python
# Process multiple images
prompts = [
    "a cat in a garden",
    "a dog on the beach",
    "a bird in the sky"
]

results = extension.batch_generate(
    prompts=prompts,
    batch_size=2,
    width=512,
    height=512
)

for i, result in enumerate(results):
    result.save(f"batch_output_{i}.png")
```

## Model Support

### Supported Model Formats
- ✅ **SafeTensors** (.safetensors)
- ✅ **GGUF** (.gguf) - Quantized models
- ✅ **FP8** - Half-precision models
- ✅ **Diffusers** - HuggingFace format
- ✅ **Checkpoint** (.ckpt) - Legacy format

### Recommended Models
- **SDXL Base**: High-quality general purpose
- **SDXL Turbo**: Fast generation
- **SD 1.5**: Lightweight option
- **Custom Fine-tunes**: Specialized styles

### Model Placement
```
forge-dream-extension/
├── models/
│   ├── checkpoints/
│   ├── lora/
│   ├── embeddings/
│   └── face_swap/
├── outputs/
└── temp/
```

## Troubleshooting

### Common Issues

#### 1. Numpy Compatibility Error
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**Solution**: Run `python fix_numpy_compatibility.py`

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce batch size
- Enable memory optimization in config
- Use lower precision (fp16)
- Enable CPU offload

#### 3. Model Loading Failed
```
Error loading model: [model_path]
```
**Solutions**:
- Check model file integrity
- Verify model format compatibility
- Ensure sufficient disk space
- Check file permissions

#### 4. Face Swap Not Working
```
InsightFace model not found
```
**Solutions**:
```bash
pip install insightface>=0.7.3
pip install onnxruntime-gpu>=1.15.0
```

### Performance Optimization

#### Memory Optimization
```json
{
  "memory_optimization": true,
  "attention_slicing": true,
  "cpu_offload": true,
  "max_memory_gb": 8
}
```

#### Speed Optimization
```json
{
  "precision": "fp16",
  "compile_model": true,
  "attention_slicing": false,
  "cpu_offload": false
}
```

### Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check memory usage
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Verify installation
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import numpy as np
print(f'NumPy: {np.__version__}')
"
```

## Updates and Maintenance

### Updating the Extension
```bash
cd forge-dream-extension
git pull origin main
pip install -r requirements.txt --upgrade
```

### Updating Dependencies
```bash
# Update PyTorch
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Update other packages
pip install --upgrade transformers diffusers accelerate
```

### Cleaning Up
```bash
# Clean pip cache
pip cache purge

# Clean temporary files
rm -rf temp/*
rm -rf outputs/temp_*
```

## Support and Community

### Getting Help
- 📖 **Documentation**: Check this guide and NUMPY_FIX_README.md
- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Join GitHub Discussions
- 🔧 **Troubleshooting**: Run diagnostic scripts

### Contributing
- Fork the repository
- Create a feature branch
- Submit pull requests
- Report issues and suggestions

### License
This project is licensed under the MIT License. See LICENSE file for details.

---

**Last Updated**: May 29, 2025  
**Version**: 2.0.0 with Numpy Compatibility Fixes  
**Tested Environments**: Windows 11, Ubuntu 22.04, macOS 13+  
**CUDA Support**: 12.8 (primary), 11.8 (fallback)
