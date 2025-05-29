# 🌟 Forge Dream Extension

**Dual-panel UI for HiDream FP8/GGUF models with integrated faceswap functionality for Stable Diffusion Forge**

## 🚀 Features

- **Dual Model Support**: Run both FP8 and GGUF HiDream models side-by-side
- **Memory Optimization**: Intelligent VRAM management for 12-24GB systems
- **Face Swap Integration**: Reactor-compatible face swapping with checkpoint support
- **Robust Error Handling**: Graceful fallbacks and comprehensive error management
- **API Support**: RESTful API endpoints for external integration

## 🔧 Recent Fixes (v1.0.0)

This version addresses critical compatibility issues with Stable Diffusion Forge:

### ✅ Fixed Issues

1. **Numpy Version Conflict**: 
   - **Problem**: Extension was installing numpy 2.x which conflicts with Forge's requirements
   - **Solution**: Constrained numpy to `>=1.24.0,<2.0.0` in requirements.txt

2. **Installation Script Robustness**:
   - **Problem**: Installation would fail completely if any dependency had conflicts
   - **Solution**: Added `--no-deps` installation strategy with selective core dependency installation
   - **Improvement**: Better error handling and graceful degradation

3. **Import Error Handling**:
   - **Problem**: Extension would crash if any module failed to import during startup
   - **Solution**: Implemented safe import system with fallback interfaces
   - **Improvement**: Extension now shows minimal interface even if some components fail

4. **Directory Creation Issues**:
   - **Problem**: Extension failed if config.json was missing or corrupted
   - **Solution**: Added fallback directory creation with sensible defaults
   - **Improvement**: More robust file system handling

5. **Forge Integration**:
   - **Problem**: Extension registration could fail silently
   - **Solution**: Enhanced error logging and graceful fallback registration
   - **Improvement**: Better compatibility with Forge's module system

## 📦 Installation

### Method 1: Git Clone (Recommended)

1. Navigate to your Forge extensions directory:
   ```bash
   cd /path/to/stable-diffusion-webui-forge/extensions/
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Valorking6/forge-dream-extension.git
   ```

3. Restart Stable Diffusion Forge

### Method 2: Manual Download

1. Download the repository as ZIP
2. Extract to `stable-diffusion-webui-forge/extensions/forge-dream-extension/`
3. Restart Stable Diffusion Forge

## 🔧 System Requirements

### Minimum Requirements
- **VRAM**: 12GB (for Fast FP8 models or Q2 GGUF)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space for models
- **Python**: 3.10+ (included with Forge)

### Recommended Requirements
- **VRAM**: 24GB (for Full FP8 models or Q6_K GGUF)
- **RAM**: 32GB system RAM
- **Storage**: 100GB+ free space
- **CUDA**: Compatible GPU (RTX 3090, 4090, A6000, etc.)

## 🎯 Model Support

### FP8 Models
- **HiDream-I1-Full**: 20GB VRAM, highest quality
- **HiDream-I1-Dev**: 16GB VRAM, development version
- **HiDream-I1-Fast**: 12GB VRAM, optimized for speed

### GGUF Models
- **Q2_K**: 7GB VRAM, lower quality but very fast
- **Q4_K_M**: 12GB VRAM, balanced quality/speed
- **Q5_K_M**: 14GB VRAM, high quality
- **Q6_K**: 15GB VRAM, near FP16 quality

### Face Swap Models
- **InSwapper**: Primary face swapping model (auto-downloaded)

## 🚀 Usage

1. **Start Forge**: Launch Stable Diffusion Forge normally
2. **Navigate to Extension**: Look for "Forge Dream" tab in the interface
3. **Download Models**: Use the model download buttons for your preferred models
4. **Generate Images**: Use either FP8 or GGUF panels for generation
5. **Enable Face Swap**: Toggle face swap for character consistency

## 🛠️ Troubleshooting

### Extension Not Loading
1. Check console for error messages
2. Ensure all dependencies are installed correctly
3. Verify Python version compatibility (3.10+)
4. Restart Forge completely

### Memory Issues
1. Monitor VRAM usage in the extension interface
2. Use smaller models (Fast FP8 or Q2 GGUF) for 12GB systems
3. Close other applications using GPU memory
4. Use the "Clear VRAM Cache" button

### Model Download Failures
1. Check internet connection
2. Verify sufficient disk space
3. Try downloading models manually if needed
4. Check Hugging Face access (some models may require authentication)

### Import Errors
The extension now handles import errors gracefully:
- Missing dependencies will show warnings but won't crash the extension
- A minimal interface will be shown if full initialization fails
- Check console logs for specific missing packages

## 🔧 Configuration

Edit `config.json` to customize:

```json
{
    "default_settings": {
        "max_vram_gb": 24,
        "default_fp8_model": "HiDream-I1-Fast",
        "default_gguf_model": "Q6_K",
        "faceswap_enabled": true
    }
}
```

## 📚 API Documentation

The extension provides REST API endpoints:

- `GET /forge_dream/models` - List available and downloaded models
- `GET /forge_dream/memory` - Get memory statistics
- `POST /forge_dream/generate` - Generate images via API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with Forge
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HiDream AI team for the amazing models
- Stable Diffusion Forge developers
- Face swap model creators
- Community contributors and testers

## 📞 Support

If you encounter issues:

1. Check this README's troubleshooting section
2. Look at console logs for specific errors
3. Create an issue on GitHub with:
   - Your system specifications
   - Forge version
   - Complete error logs
   - Steps to reproduce

---

**Note**: This extension is designed specifically for Stable Diffusion Forge. It may not work with other Stable Diffusion implementations without modifications.