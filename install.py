#!/usr/bin/env python3
"""
Enhanced Forge Dream Extension Installer with Numpy Compatibility Fixes
Handles CUDA 12.8, torch nightly, and numpy binary compatibility issues
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def fix_numpy_compatibility():
    """
    Fix numpy binary compatibility issues that cause:
    ValueError: numpy.dtype size changed, may indicate binary incompatibility
    """
    print("🔧 Applying numpy compatibility fixes...")
    
    try:
        # Force reinstall numpy with specific version constraints
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "numpy"
        ], check=False, capture_output=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", 
            "--force-reinstall", "numpy>=1.24.0,<2.0.0"
        ], check=True)
        
        # Force reinstall scipy to match numpy
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "scipy"
        ], check=False, capture_output=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", 
            "--force-reinstall", "scipy>=1.10.0,<1.12.0"
        ], check=True)
        
        print("✅ Numpy compatibility fixes applied successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to apply numpy fixes: {e}")
        return False

def install_pytorch_cuda():
    """Install PyTorch with CUDA 12.8 support"""
    print("🚀 Installing PyTorch with CUDA 12.8...")
    
    try:
        # Install PyTorch nightly with CUDA 12.8
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--pre", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/nightly/cu128"
        ], check=True)
        
        print("✅ PyTorch with CUDA 12.8 installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_xformers():
    """Install xformers for memory optimization"""
    print("⚡ Installing xformers...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "xformers>=0.0.20"
        ], check=True)
        
        print("✅ xformers installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install xformers: {e}")
        return False

def install_requirements():
    """Install all requirements with compatibility handling"""
    print("📦 Installing requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_insightface():
    """Setup InsightFace for face swapping"""
    print("👤 Setting up InsightFace...")
    
    try:
        # Install InsightFace with ONNX GPU support
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "insightface>=0.7.3", "onnxruntime-gpu>=1.15.0"
        ], check=True)
        
        print("✅ InsightFace setup completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup InsightFace: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("🔍 Verifying installation...")
    
    try:
        # Test numpy
        import numpy as np
        print(f"✅ Numpy {np.__version__} working")
        
        # Test torch
        import torch
        print(f"✅ PyTorch {torch.__version__} working")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
        
        # Test other key components
        import scipy
        print(f"✅ SciPy {scipy.__version__} working")
        
        try:
            import xformers
            print(f"✅ xformers {xformers.__version__} working")
        except ImportError:
            print("⚠️ xformers not available (optional)")
        
        try:
            import insightface
            print("✅ InsightFace available")
        except ImportError:
            print("⚠️ InsightFace not available (optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_config():
    """Create default configuration"""
    print("⚙️ Creating configuration...")
    
    config = {
        "model_path": "./models",
        "output_path": "./outputs",
        "cuda_enabled": True,
        "memory_optimization": True,
        "face_swap_enabled": True,
        "numpy_fix_applied": True
    }
    
    config_file = Path(__file__).parent / "config.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def main():
    """Main installation process"""
    print("🎨 Forge Dream Extension Installer")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Apply numpy compatibility fixes first
    if not fix_numpy_compatibility():
        print("❌ Critical: Numpy compatibility fixes failed")
        sys.exit(1)
    
    # Install PyTorch with CUDA
    if not install_pytorch_cuda():
        print("❌ Critical: PyTorch installation failed")
        sys.exit(1)
    
    # Install xformers
    install_xformers()  # Non-critical
    
    # Install requirements
    if not install_requirements():
        print("❌ Critical: Requirements installation failed")
        sys.exit(1)
    
    # Setup InsightFace
    setup_insightface()  # Non-critical
    
    # Create configuration
    create_config()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("📖 Check NUMPY_FIX_README.md for troubleshooting")
        print("📋 See INSTALLATION_SUMMARY.md for usage instructions")
    else:
        print("\n⚠️ Installation completed with warnings")
        print("🔧 Run fix_numpy_compatibility.py if you encounter issues")

if __name__ == "__main__":
    main()
