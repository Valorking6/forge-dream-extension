# 🌟 Forge Dream Extension

A comprehensive Stable Diffusion Forge extension that provides a dual-panel interface for HiDream FP8/GGUF models with integrated Reactor-compatible faceswap functionality.

## ✨ Features

### 🚀 Dual Model Support
- **FP8 Models**: High-quality HiDream models optimized for 12-24GB VRAM
- **GGUF Models**: Quantized models for efficient inference (7-15GB VRAM)
- **Automatic Model Management**: Download and switch between models seamlessly

### 🎭 Integrated FaceSwap
- **Reactor Compatibility**: Full compatibility with Reactor faceswap models
- **Face Checkpoints**: Create and manage reusable face embeddings
- **Batch Processing**: Process multiple images with consistent face swapping
- **Advanced Controls**: Fine-tune detection thresholds and blending ratios

### 🖥️ Dual-Panel Interface
- **Side-by-Side Layout**: FP8 models on the left, GGUF models on the right
- **Text2Img & Img2Img**: Both generation modes available in each panel
- **Memory Monitoring**: Real-time VRAM usage tracking
- **Responsive Design**: Optimized for different screen sizes

### ⚡ Performance Optimization
- **Memory Management**: Automatic VRAM optimization for 12-24GB systems
- **Model Offloading**: Dynamic loading/unloading based on memory pressure
- **Batch Size Adjustment**: Automatic batch size optimization
- **Mixed Precision**: FP8/FP16/BF16 support for optimal performance

## 📋 Requirements

### System Requirements
- **VRAM**: 12-24GB recommended (minimum 8GB)
- **CUDA**: CUDA 12.4+ recommended for Flash Attention
- **Python**: 3.8+ with PyTorch 2.3.1+
- **Stable Diffusion Forge**: Latest version

### Model Requirements
- **FP8 Models**: 12-20GB VRAM depending on variant
- **GGUF Models**: 7-15GB VRAM depending on quantization
- **FaceSwap Models**: Additional ~500MB VRAM

## 🚀 Installation

### Method 1: Automatic Installation

1. **Clone the extension**:
   ```bash
   cd /path/to/stable-diffusion-forge/extensions
   git clone https://github.com/Valorking6/forge-dream-extension.git
   ```

2. **Run the installer**:
   ```bash
   cd forge-dream-extension
   python install.py
   ```

3. **Restart Stable Diffusion Forge**

### Method 2: Manual Installation

1. **Download and extract** the extension to your Forge extensions directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create model directories**:
   ```bash
   mkdir -p models/hidream/{fp8,gguf}
   mkdir -p models/faceswaplab/faces
   ```

4. **Download essential models**:
   - Place `inswapper_128.onnx` in `models/faceswaplab/`
   - HiDream models will be downloaded automatically

5. **Restart Forge**

## 📖 Usage Guide

### Getting Started

1. **Launch Forge** and navigate to the "Forge Dream" tab
2. **Monitor VRAM** usage in the top indicator
3. **Select models** from the dropdowns (download if needed)
4. **Choose generation mode** (Text2Img or Img2Img)
5. **Configure parameters** and generate images

### Model Selection

#### FP8 Models (Left Panel)
- **HiDream-I1-Full**: Highest quality, requires 20GB VRAM
- **HiDream-I1-Dev**: Balanced quality/speed, requires 16GB VRAM  
- **HiDream-I1-Fast**: Fastest generation, requires 12GB VRAM

#### GGUF Models (Right Panel)
- **Q6_K**: Near FP16 quality, requires 15GB VRAM
- **Q5**: High quality, requires 14GB VRAM
- **Q4**: Balanced quality/efficiency, requires 12GB VRAM
- **Q2**: Lower quality, very efficient, requires 7GB VRAM

### FaceSwap Configuration

1. **Enable FaceSwap** in the accordion section
2. **Upload source image** or select a face checkpoint
3. **Adjust detection thresholds** for optimal face detection
4. **Set face indices** to target specific faces
5. **Configure blend ratio** for natural-looking results

### Creating Face Checkpoints

1. **Click "Create Checkpoint"** in any faceswap section
2. **Provide a name** for the checkpoint
3. **Upload multiple images** of the same person
4. **Click "Create"** to build the checkpoint
5. **Select the checkpoint** from the dropdown for future use

### Memory Management

- **Monitor VRAM** usage in the top indicator
- **Green**: Low usage (< 50%)
- **Yellow**: Medium usage (50-75%)
- **Orange**: High usage (75-90%)
- **Red**: Critical usage (> 90%)

The extension automatically:
- Adjusts batch sizes based on available memory
- Unloads models when memory is low
- Optimizes precision settings for your hardware

## ⚙️ Configuration

### Config Files

#### `config.json`
```json
{
    "default_settings": {
        "max_vram_gb": 24,
        "auto_download_models": true,
        "default_fp8_model": "HiDream-I1-Fast",
        "default_gguf_model": "Q6_K",
        "faceswap_enabled": true,
        "batch_size": 1,
        "inference_steps": 28,
        "guidance_scale": 0.0
    }
}
```

#### `model_urls.json`
Contains download URLs and specifications for all supported models.

### Environment Variables

- `FORGE_DREAM_MAX_VRAM`: Override maximum VRAM limit
- `FORGE_DREAM_CACHE_DIR`: Custom cache directory for models
- `FORGE_DREAM_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## 🔧 API Reference

### REST Endpoints

#### Get Models
```http
GET /forge_dream/models
```
Returns available and downloaded models.

#### Memory Stats
```http
GET /forge_dream/memory
```
Returns current VRAM usage statistics.

#### Generate Image
```http
POST /forge_dream/generate
Content-Type: application/json

{
    "model_type": "fp8",
    "model_name": "HiDream-I1-Fast",
    "prompt": "A beautiful landscape",
    "parameters": {
        "width": 512,
        "height": 512,
        "num_inference_steps": 28,
        "guidance_scale": 0.0
    }
}
```

### JavaScript API

```javascript
// Access extension state
window.ForgeDream.extensionState

// Refresh models
window.ForgeDream.refreshModels()

// Clear memory cache
window.ForgeDream.clearMemoryCache()

// Show notification
window.ForgeDream.showNotification("Message", "success")
```

## 🐛 Troubleshooting

### Common Issues

#### "Failed to load model"
- **Check VRAM**: Ensure sufficient VRAM is available
- **Verify download**: Check if model files are completely downloaded
- **Restart Forge**: Sometimes a restart resolves loading issues

#### "No faces detected"
- **Adjust threshold**: Lower the detection threshold
- **Check image quality**: Ensure faces are clearly visible
- **Try different angles**: Profile shots may not work well

#### "Out of memory" errors
- **Reduce batch size**: Lower the number of images per generation
- **Use smaller models**: Switch to Q4 or Q2 GGUF variants
- **Close other applications**: Free up system memory

#### Extension not appearing
- **Check installation**: Verify all files are in the correct location
- **Install dependencies**: Run `pip install -r requirements.txt`
- **Check logs**: Look for error messages in the Forge console

### Performance Optimization

#### For 12GB VRAM Systems
- Use HiDream-I1-Fast or Q4 GGUF models
- Set batch size to 1
- Enable model offloading in config
- Close unnecessary browser tabs

#### For 16GB VRAM Systems
- Use HiDream-I1-Dev or Q5 GGUF models
- Batch size 1-2 depending on resolution
- Monitor memory usage closely

#### For 24GB VRAM Systems
- Use any model variant
- Batch sizes up to 4 for most models
- Can run both panels simultaneously

### Debug Mode

Enable debug logging by setting:
```bash
export FORGE_DREAM_LOG_LEVEL=DEBUG
```

This provides detailed information about:
- Model loading processes
- Memory allocation
- Face detection results
- Generation parameters

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork the repository**
2. **Create a development branch**
3. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Make your changes**
5. **Run tests**:
   ```bash
   python -m pytest tests/
   ```
6. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HiDream Team** for the amazing HiDream models
- **Reactor Team** for the faceswap technology
- **Stable Diffusion Forge** for the excellent framework
- **Community Contributors** for feedback and improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Valorking6/forge-dream-extension/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Valorking6/forge-dream-extension/discussions)
- **Discord**: [Join our Discord](https://discord.gg/your-server)

---

**Made with ❤️ for the Stable Diffusion community**
