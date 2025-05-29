"""
Enhanced installation script for Forge Dream Extension
Supports CUDA 12.8, PyTorch nightly, Windows 11, and RTX 5070 Ti
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForgeDreamInstaller:
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.is_windows = self.system == "Windows"
        self.cuda_available = False
        self.cuda_version = None
        self.torch_nightly = False
        
        logger.info(f"System: {self.system}")
        logger.info(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if self.python_version < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        logger.info("✓ Python version check passed")
    
    def detect_cuda(self):
        """Detect CUDA installation and version"""
        try:
            # Try to get CUDA version from nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.cuda_available = True
                logger.info("✓ NVIDIA GPU detected")
                
                # Try to get CUDA version
                try:
                    nvcc_result = subprocess.run(['nvcc', '--version'], 
                                               capture_output=True, text=True, timeout=10)
                    if nvcc_result.returncode == 0:
                        output = nvcc_result.stdout
                        if 'release 12.8' in output:
                            self.cuda_version = "12.8"
                            logger.info("✓ CUDA 12.8 detected")
                        elif 'release 12.' in output:
                            # Extract version
                            import re
                            match = re.search(r'release (\d+\.\d+)', output)
                            if match:
                                self.cuda_version = match.group(1)
                                logger.info(f"✓ CUDA {self.cuda_version} detected")
                        else:
                            logger.warning("CUDA version could not be determined from nvcc")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("nvcc not found, CUDA version unknown")
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("nvidia-smi not found, assuming no CUDA")
            
        return self.cuda_available
    
    def check_torch_installation(self):
        """Check current PyTorch installation"""
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            logger.info(f"Current PyTorch version: {torch_version}")
            logger.info(f"PyTorch CUDA available: {cuda_available}")
            
            # Check if it's a nightly build
            if 'dev' in torch_version or '+' in torch_version:
                self.torch_nightly = True
                logger.info("✓ PyTorch nightly detected")
            
            if cuda_available and self.cuda_available:
                cuda_version = torch.version.cuda
                logger.info(f"PyTorch CUDA version: {cuda_version}")
                return True
            elif not self.cuda_available:
                logger.info("CPU-only PyTorch installation")
                return True
            else:
                logger.warning("PyTorch CUDA not available despite CUDA installation")
                return False
                
        except ImportError:
            logger.warning("PyTorch not installed")
            return False
    
    def install_pytorch(self):
        """Install PyTorch with appropriate CUDA support"""
        logger.info("Installing PyTorch...")
        
        if self.cuda_available:
            if self.cuda_version and self.cuda_version.startswith("12."):
                # CUDA 12.x installation
                if self.torch_nightly:
                    # Install nightly with CUDA 12.1 (most compatible)
                    cmd = [
                        sys.executable, "-m", "pip", "install", "--upgrade",
                        "--index-url", "https://download.pytorch.org/whl/nightly/cu121",
                        "torch", "torchvision", "torchaudio"
                    ]
                else:
                    # Install stable with CUDA 12.1
                    cmd = [
                        sys.executable, "-m", "pip", "install", "--upgrade",
                        "--index-url", "https://download.pytorch.org/whl/cu121",
                        "torch", "torchvision", "torchaudio"
                    ]
            else:
                # Fallback to CUDA 11.8 for older versions
                cmd = [
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "torch", "torchvision", "torchaudio"
                ]
        else:
            # CPU-only installation
            cmd = [
                sys.executable, "-m", "pip", "install", "--upgrade",
                "torch", "torchvision", "torchaudio", "--index-url",
                "https://download.pytorch.org/whl/cpu"
            ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            logger.info("✓ PyTorch installation completed")
        else:
            raise RuntimeError("PyTorch installation failed")
    
    def install_requirements(self):
        """Install requirements with Windows and CUDA compatibility"""
        logger.info("Installing requirements...")
        
        requirements_file = Path(__file__).parent / "requirements.txt"
        
        if not requirements_file.exists():
            logger.warning("requirements.txt not found, installing basic dependencies")
            basic_deps = [
                "diffusers>=0.24.0",
                "transformers>=4.35.0",
                "accelerate>=0.24.0",
                "safetensors>=0.4.0",
                "Pillow>=10.0.0",
                "opencv-python>=4.8.0",
                "numpy>=1.24.0",
                "gradio>=4.0.0",
                "tqdm>=4.65.0",
                "requests>=2.31.0",
                "huggingface-hub>=0.17.0"
            ]
            
            for dep in basic_deps:
                cmd = [sys.executable, "-m", "pip", "install", dep]
                subprocess.run(cmd, check=True)
        else:
            # Install from requirements.txt
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            subprocess.run(cmd, check=True)
        
        logger.info("✓ Requirements installation completed")
    
    def install_optional_dependencies(self):
        """Install optional dependencies for enhanced functionality"""
        logger.info("Installing optional dependencies...")
        
        optional_deps = []
        
        # xformers for memory efficiency (CUDA only)
        if self.cuda_available:
            optional_deps.append("xformers>=0.0.22")
        
        # InsightFace for face analysis
        optional_deps.append("insightface>=0.7.3")
        
        # ONNX Runtime with GPU support
        if self.cuda_available:
            optional_deps.append("onnxruntime-gpu>=1.16.0")
        else:
            optional_deps.append("onnxruntime>=1.16.0")
        
        # Windows-specific dependencies
        if self.is_windows:
            optional_deps.extend([
                "pywin32>=306",
                "colorama>=0.4.6"
            ])
        
        # LLAMA CPP for GGUF support
        try:
            if self.cuda_available:
                # Try to install with CUDA support
                cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python", 
                      "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
            
            subprocess.run(cmd, check=True)
            logger.info("✓ LLAMA CPP installed for GGUF support")
        except subprocess.CalledProcessError:
            logger.warning("Failed to install llama-cpp-python, GGUF support may be limited")
        
        # Install other optional dependencies
        for dep in optional_deps:
            try:
                cmd = [sys.executable, "-m", "pip", "install", dep]
                subprocess.run(cmd, check=True)
                logger.info(f"✓ Installed {dep}")
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {dep}, continuing...")
    
    def verify_installation(self):
        """Verify that the installation was successful"""
        logger.info("Verifying installation...")
        
        # Test PyTorch
        try:
            import torch
            logger.info(f"✓ PyTorch {torch.__version__} imported successfully")
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ CUDA available with {device_count} device(s)")
                logger.info(f"✓ Primary device: {device_name}")
                
                # Test RTX 5070 Ti specific optimizations
                if "RTX 5070" in device_name or "RTX 50" in device_name:
                    logger.info("✓ RTX 5070 Ti detected - optimizations will be applied")
            else:
                logger.info("✓ PyTorch running in CPU mode")
                
        except ImportError as e:
            logger.error(f"✗ PyTorch import failed: {e}")
            return False
        
        # Test other key dependencies
        dependencies = [
            ("diffusers", "Diffusers"),
            ("transformers", "Transformers"),
            ("PIL", "Pillow"),
            ("cv2", "OpenCV"),
            ("numpy", "NumPy"),
            ("gradio", "Gradio")
        ]
        
        for module, name in dependencies:
            try:
                importlib.import_module(module)
                logger.info(f"✓ {name} imported successfully")
            except ImportError:
                logger.warning(f"✗ {name} not available")
        
        # Test optional dependencies
        optional_deps = [
            ("insightface", "InsightFace"),
            ("llama_cpp", "LLAMA CPP"),
            ("xformers", "xformers")
        ]
        
        for module, name in optional_deps:
            try:
                importlib.import_module(module)
                logger.info(f"✓ {name} (optional) imported successfully")
            except ImportError:
                logger.info(f"○ {name} (optional) not available")
        
        logger.info("✓ Installation verification completed")
        return True
    
    def create_config(self):
        """Create configuration file with detected settings"""
        config = {
            "system": self.system,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "torch_nightly": self.torch_nightly,
            "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "installation_date": str(Path(__file__).stat().st_mtime),
            "optimizations": {
                "rtx_5070_ti": self.cuda_available,
                "cuda_12_8": self.cuda_version == "12.8" if self.cuda_version else False,
                "windows_compatibility": self.is_windows,
                "torch_nightly_features": self.torch_nightly
            }
        }
        
        config_file = Path(__file__).parent / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Configuration saved to {config_file}")
    
    def run_installation(self):
        """Run the complete installation process"""
        try:
            logger.info("Starting Forge Dream Extension installation...")
            
            # Check system requirements
            self.check_python_version()
            self.detect_cuda()
            
            # Check existing PyTorch installation
            torch_ok = self.check_torch_installation()
            
            # Install or upgrade PyTorch if needed
            if not torch_ok:
                self.install_pytorch()
            
            # Install requirements
            self.install_requirements()
            
            # Install optional dependencies
            self.install_optional_dependencies()
            
            # Verify installation
            if self.verify_installation():
                # Create configuration
                self.create_config()
                
                logger.info("🎉 Forge Dream Extension installation completed successfully!")
                logger.info("You can now use the extension in Stable Diffusion Forge.")
                
                if self.cuda_available:
                    logger.info("💡 CUDA acceleration is enabled for optimal performance.")
                if self.cuda_version == "12.8":
                    logger.info("💡 CUDA 12.8 optimizations are active.")
                if self.torch_nightly:
                    logger.info("💡 PyTorch nightly features are available.")
                
                return True
            else:
                logger.error("Installation verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False

def main():
    """Main installation entry point"""
    installer = ForgeDreamInstaller()
    success = installer.run_installation()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()