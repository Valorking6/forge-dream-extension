# CUDA 12.8 Installation Guide for Forge Dream Extension

This guide provides specific instructions for installing the Forge Dream Extension with CUDA 12.8, PyTorch nightly, and RTX 5070 Ti support on Windows 11.

## Prerequisites

### System Requirements
- **OS**: Windows 11 (64-bit)
- **GPU**: RTX 5070 Ti or compatible NVIDIA GPU
- **CUDA**: Version 12.8
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### CUDA 12.8 Installation

1. **Download CUDA 12.8**
   ```
   https://developer.nvidia.com/cuda-12-8-0-download-archive
   ```

2. **Install CUDA Toolkit**
   - Run the installer as Administrator
   - Choose "Custom" installation
   - Select:
     - CUDA Toolkit
     - CUDA Samples (optional)
     - CUDA Documentation (optional)
   - Uncheck Visual Studio Integration if not needed

3. **Verify CUDA Installation**
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### NVIDIA Driver Requirements
- **Minimum Driver Version**: 546.01 or higher
- **Recommended**: Latest Game Ready or Studio drivers

## Installation Steps

### Method 1: Automatic Installation (Recommended)

1. **Clone the Repository**
   ```cmd
   git clone https://github.com/Valorking6/forge-dream-extension.git
   cd forge-dream-extension
   ```

2. **Run the Enhanced Installer**
   ```cmd
   python install.py
   ```
   
   The installer will automatically:
   - Detect your CUDA 12.8 installation
   - Install PyTorch with CUDA 12.1 support (most compatible)
   - Configure RTX 5070 Ti optimizations
   - Install all required dependencies

### Method 2: Manual Installation

1. **Install PyTorch with CUDA Support**
   
   For stable PyTorch:
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   
   For PyTorch nightly (advanced users):
   ```cmd
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
   ```

2. **Install Core Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Install Optional Dependencies**
   ```cmd
   # For memory optimization
   pip install xformers
   
   # For face analysis
   pip install insightface
   pip install onnxruntime-gpu
   
   # For GGUF model support
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```

## RTX 5070 Ti Optimizations

The extension automatically applies these optimizations for RTX 5070 Ti:

### Memory Optimizations
- **Memory Fraction**: 90% GPU memory allocation
- **Memory Pool**: Enabled with 512MB split size
- **CPU Offload**: Automatic for large models

### Performance Optimizations
- **cuDNN Benchmark**: Enabled for consistent input sizes
- **TensorFloat-32**: Enabled for faster training/inference
- **Mixed Precision**: bfloat16 support (PyTorch nightly)
- **Xformers**: Memory-efficient attention mechanisms

### CUDA 12.8 Specific Features
- **Enhanced Memory Management**: Better allocation strategies
- **Improved Kernel Fusion**: Reduced memory transfers
- **Advanced Scheduling**: Better GPU utilization

## Verification

### Test CUDA Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

Expected output for RTX 5070 Ti:
```
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
Device name: NVIDIA GeForce RTX 5070 Ti
```

### Test Extension Loading
```python
from scripts.forge_dream import initialize_extension
success = initialize_extension()
print(f"Extension initialized: {success}")
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Add to environment variables
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

#### 2. PyTorch Not Detecting CUDA
- Verify NVIDIA drivers are up to date
- Reinstall PyTorch with correct CUDA version
- Check Windows PATH includes CUDA bin directory

#### 3. xformers Installation Fails
```cmd
# Install pre-compiled version
pip install xformers --index-url https://download.pytorch.org/whl/cu121
```

#### 4. InsightFace Installation Issues
```cmd
# Install dependencies first
pip install onnx onnxruntime-gpu
pip install insightface --no-deps
pip install opencv-python scikit-image
```

### Performance Tuning

#### For Maximum Performance
```python
# Add to your environment
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### For Stability (if experiencing crashes)
```python
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Disable some optimizations
export TORCH_CUDNN_BENCHMARK=0
```

## Model Compatibility

### Supported Model Formats
- **GGUF**: Quantized models (4-bit, 8-bit)
- **FP8**: Half-precision models
- **SafeTensors**: Standard format
- **Diffusers**: HuggingFace format

### Recommended Models for RTX 5070 Ti
- **SDXL**: Full resolution (1024x1024)
- **SD 1.5**: High resolution (768x768+)
- **GGUF Q4**: Efficient quantized models
- **FP8**: Balanced quality/performance

## Advanced Configuration

### Custom CUDA Settings
Create `cuda_config.json`:
```json
{
  "memory_fraction": 0.9,
  "allow_tf32": true,
  "benchmark": true,
  "deterministic": false,
  "max_split_size_mb": 512
}
```

### Environment Variables
```cmd
# Windows batch file (run_forge_dream.bat)
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TORCH_CUDNN_V8_API_ENABLED=1
python webui.py --extension forge-dream-extension
```

## Support

### Getting Help
1. Check the [Issues](https://github.com/Valorking6/forge-dream-extension/issues) page
2. Verify your installation with the verification steps
3. Include system information when reporting issues:
   - Windows version
   - CUDA version
   - PyTorch version
   - GPU model
   - Error messages

### Useful Commands
```cmd
# System information
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Extension status
python -c "from scripts.forge_dream import get_extension_info; print(get_extension_info())"
```

## Updates

The extension will automatically detect and optimize for:
- New CUDA versions
- PyTorch updates
- Driver improvements
- Hardware changes

To update:
```cmd
git pull origin main
python install.py
```

---

**Note**: This installation guide is specifically optimized for CUDA 12.8 and RTX 5070 Ti. For other configurations, the automatic installer will detect and adapt accordingly.