#!/usr/bin/env python3
"""
Installation script for Forge Dream extension
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install required packages with proper error handling"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("Installing requirements...")
        try:
            # Use --no-deps to avoid conflicts, then install dependencies separately
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file),
                "--no-deps"
            ])
            
            # Install core dependencies that are compatible with Forge
            core_deps = [
                "numpy>=1.24.0,<2.0.0",
                "Pillow>=10.0.0",
                "requests>=2.31.0",
                "tqdm>=4.65.0"
            ]
            
            for dep in core_deps:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", dep, "--upgrade"
                    ])
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Failed to install {dep}: {e}")
                    
        except subprocess.CalledProcessError as e:
            print(f"Warning: Some requirements may not have installed correctly: {e}")
            print("The extension will attempt to use existing packages.")
    else:
        print("Requirements file not found!")

def create_directories():
    """Create necessary directories"""
    base_path = Path(__file__).parent
    config_file = base_path / "config.json"
    
    # Create basic directories even if config doesn't exist
    default_dirs = ["models/fp8", "models/gguf", "models/faceswap", "temp"]
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for dir_name, dir_path in config.get("model_directories", {}).items():
                full_path = base_path / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {full_path}")
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")
            print("Creating default directories...")
    
    # Create default directories
    for dir_path in default_dirs:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def download_essential_models():
    """Download essential models for basic functionality"""
    try:
        # Only try to import if we're not in installation phase
        sys.path.insert(0, str(Path(__file__).parent))
        from scripts.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Download inswapper model for faceswap
        print("Downloading essential faceswap model...")
        manager.download_faceswap_model("inswapper")
        
        print("Installation completed successfully!")
        print("You can now use the Forge Dream extension.")
        
    except ImportError as e:
        print("Model manager not available during installation.")
        print("Models will be downloaded when the extension is first used.")
    except Exception as e:
        print(f"Warning: Could not download models during installation: {e}")
        print("Models will be downloaded when the extension is first used.")

def main():
    """Main installation function"""
    print("Installing Forge Dream Extension...")
    
    try:
        install_requirements()
        create_directories()
        download_essential_models()
        
        print("\n" + "="*50)
        print("Forge Dream Extension installed successfully!")
        print("Please restart Stable Diffusion Forge to use the extension.")
        print("="*50)
        
    except Exception as e:
        print(f"Installation completed with warnings: {e}")
        print("The extension should still work, but some features may be limited.")
        print("Please restart Stable Diffusion Forge to use the extension.")

if __name__ == "__main__":
    main()