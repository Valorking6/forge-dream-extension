# Forge Dream Extension

A powerful extension for Stable Diffusion Forge that provides a dual-panel UI for HiDream FP8/GGUF models with integrated faceswap capabilities. Now with full Windows 11, CUDA 12.8, PyTorch nightly, and RTX 5070 Ti support!

## ✨ Features

### 🎨 Dual-Panel Interface
- **Left Panel**: Model loading and configuration
- **Right Panel**: Image generation with advanced controls
- **Bottom Panel**: Integrated face swap functionality

### 🚀 Model Support
- **GGUF Models**: Quantized models with llama-cpp-python integration
- **FP8 Models**: Half-precision models for memory efficiency
- **SafeTensors**: Standard diffusion model format
- **Diffusers**: HuggingFace pipeline format

### 🔧 Advanced Features
- **Face Swap**: Integrated InsightFace-powered face swapping
- **Memory Optimization**: Smart memory management for large models
- **CUDA Acceleration**: Full CUDA 12.8 support with RTX optimizations
- **Cross-Platform**: Windows 11, Linux, and macOS support

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 5GB free space

### Recommended for Optimal Performance
- **OS**: Windows 11 (64-bit)
- **GPU**: RTX 5070 Ti or RTX 40/30 series
- **CUDA**: 12.8 (automatically detected)
- **RAM**: 32GB
- **Storage**: NVMe SSD with 20GB+ free space

### Supported Hardware
- **NVIDIA GPUs**: RTX 50/40/30/20 series, GTX 16 series
- **AMD GPUs**: Limited support via ROCm
- **Intel GPUs**: Experimental support
- **CPU**: Fallback support for all processors

## 🚀 Quick Installation

### Automatic Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Valorking6/forge-dream-extension.git
   cd forge-dream-extension
   ```

2. **Run the enhanced installer**:
   ```bash
   python install.py
   ```

The installer will automatically:
- ✅ Detect your system configuration
- ✅ Install PyTorch with appropriate CUDA support
- ✅ Configure RTX 5070 Ti optimizations
- ✅ Install all required dependencies
- ✅ Verify the installation

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

1. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 12.x (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For PyTorch nightly (advanced users)
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
   ```

2. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install optional dependencies**:
   ```bash
   # Memory optimization
   pip install xformers
   
   # Face analysis
   pip install insightface onnxruntime-gpu
   
   # GGUF support
   pip install llama-cpp-python
   ```

</details>

## 📋 Installation for Specific Setups

### CUDA 12.8 + RTX 5070 Ti + Windows 11
See our detailed [CUDA 12.8 Installation Guide](INSTALL_CUDA_12_8.md) for step-by-step instructions.

### PyTorch Nightly
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python install.py
```

### CPU-Only Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python install.py
```

## 🎯 Usage

### In Stable Diffusion Forge

1. **Install the extension** in your Forge extensions directory
2. **Restart Forge**
3. **Navigate** to the "Forge Dream" tab
4. **Load a model** using the model path input
5. **Generate images** with the intuitive interface

### Standalone Usage

```python
from scripts.forge_dream import create_interface

# Launch the interface
interface = create_interface()
interface.launch()
```

## 🔧 Configuration

### Automatic Configuration
The extension automatically detects and configures:
- ✅ CUDA version and capabilities
- ✅ GPU model and optimizations
- ✅ Available memory
- ✅ PyTorch features

### Manual Configuration
Create a `config.json` file for custom settings:

```json
{
  "device": "cuda",
  "dtype": "float16",
  "memory_fraction": 0.9,
  "enable_xformers": true,
  "enable_cpu_offload": true,
  "rtx_optimizations": true
}
```

## 🎨 Model Loading

### Supported Formats

#### GGUF Models
```python
# Quantized models for memory efficiency
model_path = "path/to/model.gguf"
model_type = "gguf"
```

#### FP8 Models
```python
# Half-precision models
model_path = "path/to/model.fp8"
model_type = "fp8"
```

#### Standard Models
```python
# SafeTensors and Diffusers format
model_path = "path/to/model.safetensors"
model_type = "standard"
```

### Model Recommendations

#### For RTX 5070 Ti (16GB VRAM)
- **SDXL**: Full resolution models
- **GGUF Q4**: 4-bit quantized for efficiency
- **FP8**: Balanced quality/performance

#### For RTX 4060 Ti (8GB VRAM)
- **SD 1.5**: Standard resolution
- **GGUF Q8**: 8-bit quantized
- **FP8**: With CPU offload

#### For RTX 3060 (12GB VRAM)
- **SD 1.5**: Recommended
- **GGUF Q4**: Best performance
- **CPU Offload**: Enabled

## 🎭 Face Swap Feature

### Requirements
- **InsightFace**: For face detection and analysis
- **OpenCV**: For image processing
- **ONNX Runtime**: For GPU acceleration

### Usage
1. **Load source image**: The face you want to use
2. **Load target image**: The image to modify
3. **Click "Perform Face Swap"**
4. **Download result**: High-quality face-swapped image

### Supported Formats
- **Input**: JPG, PNG, WebP, BMP
- **Output**: PNG (high quality)
- **Resolution**: Up to 4K (limited by VRAM)

## ⚡ Performance Optimizations

### RTX 5070 Ti Specific
- **Memory Management**: 90% VRAM allocation
- **TensorFloat-32**: Enabled for faster inference
- **Mixed Precision**: bfloat16 support
- **Xformers**: Memory-efficient attention

### CUDA 12.8 Features
- **Enhanced Memory Pool**: Better allocation
- **Kernel Fusion**: Reduced memory transfers
- **Advanced Scheduling**: Improved GPU utilization

### General Optimizations
- **Model CPU Offload**: For large models
- **Gradient Checkpointing**: Memory savings
- **Attention Slicing**: Reduced memory usage

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Set environment variable
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### PyTorch Not Detecting CUDA
1. Update NVIDIA drivers
2. Reinstall PyTorch with correct CUDA version
3. Verify CUDA installation with `nvcc --version`

#### Extension Not Loading
1. Check Python version (3.8+ required)
2. Verify all dependencies are installed
3. Check Forge logs for error messages

#### Face Swap Not Working
1. Install InsightFace: `pip install insightface`
2. Install ONNX Runtime GPU: `pip install onnxruntime-gpu`
3. Verify face detection in source images

### Performance Issues

#### Slow Generation
- Enable xformers: `pip install xformers`
- Use FP16 precision
- Enable model CPU offload
- Reduce image resolution

#### Memory Issues
- Lower memory fraction in config
- Enable CPU offload
- Use quantized (GGUF) models
- Close other applications

## 🔄 Updates

### Automatic Updates
The extension checks for updates and optimizations:
- New CUDA versions
- PyTorch improvements
- Driver updates
- Hardware changes

### Manual Updates
```bash
git pull origin main
python install.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Valorking6/forge-dream-extension.git
cd forge-dream-extension
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion Forge**: Base framework
- **HuggingFace**: Diffusers library
- **InsightFace**: Face analysis
- **llama.cpp**: GGUF model support
- **PyTorch Team**: Deep learning framework

## 📞 Support

### Getting Help
- 📖 [Documentation](https://github.com/Valorking6/forge-dream-extension/wiki)
- 🐛 [Issue Tracker](https://github.com/Valorking6/forge-dream-extension/issues)
- 💬 [Discussions](https://github.com/Valorking6/forge-dream-extension/discussions)

### System Information for Bug Reports
```bash
# Run this command and include output in bug reports
python -c "
import torch, platform, sys
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Valorking6/forge-dream-extension&type=Date)](https://star-history.com/#Valorking6/forge-dream-extension&Date)

---

**Made with ❤️ for the Stable Diffusion community**

*Compatible with Windows 11, CUDA 12.8, PyTorch nightly, and RTX 5070 Ti*