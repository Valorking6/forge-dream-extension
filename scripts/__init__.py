"""
Forge Dream Extension - Windows and CUDA 12.8 Compatible
Dual-panel UI for HiDream FP8/GGUF models with integrated faceswap
"""

import os
import sys
import platform
from pathlib import Path

# Windows compatibility fixes
if platform.system() == "Windows":
    # Ensure proper path handling on Windows
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
    
    # Add current directory to Python path for imports
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

# Version info
__version__ = "1.1.0"
__author__ = "Forge Dream Team"
__description__ = "Forge Dream Extension with Windows and CUDA 12.8 compatibility"

# Extension metadata
EXTENSION_NAME = "forge-dream-extension"
EXTENSION_VERSION = __version__
EXTENSION_DESCRIPTION = __description__

# Compatibility flags
WINDOWS_COMPATIBLE = True
CUDA_12_8_COMPATIBLE = True
TORCH_NIGHTLY_COMPATIBLE = True
RTX_5070_TI_OPTIMIZED = True

def get_extension_info():
    """Get extension information"""
    return {
        "name": EXTENSION_NAME,
        "version": EXTENSION_VERSION,
        "description": EXTENSION_DESCRIPTION,
        "windows_compatible": WINDOWS_COMPATIBLE,
        "cuda_12_8_compatible": CUDA_12_8_COMPATIBLE,
        "torch_nightly_compatible": TORCH_NIGHTLY_COMPATIBLE,
        "rtx_5070_ti_optimized": RTX_5070_TI_OPTIMIZED
    }

# Import main module with error handling
try:
    from .forge_dream import on_ui_tabs, initialize_extension
    
    # Auto-initialize if in Forge environment
    if hasattr(sys, 'modules') and any('forge' in name.lower() for name in sys.modules.keys()):
        initialize_extension()
        
except ImportError as e:
    print(f"Warning: Failed to import forge_dream module: {e}")
    print("This may be normal if dependencies are not yet installed.")
    
    # Provide fallback functions
    def on_ui_tabs():
        import gradio as gr
        error_msg = f"Forge Dream Extension failed to load: {e}"
        return [(gr.HTML(f"<h3>{error_msg}</h3>"), "Forge Dream (Error)", "forge_dream")]
    
    def initialize_extension():
        return False

# Export main functions
__all__ = ["on_ui_tabs", "initialize_extension", "get_extension_info"]