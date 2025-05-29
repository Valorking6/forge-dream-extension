#!/usr/bin/env python3
"""
Standalone Numpy Compatibility Fix Script for Forge Dream Extension
Resolves: ValueError: numpy.dtype size changed, may indicate binary incompatibility

This script can be run independently to fix numpy/scipy compatibility issues
that occur with CUDA 12.8, torch nightly, and various package combinations.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def print_header():
    """Print script header"""
    print("🔧 Numpy Compatibility Fix for Forge Dream Extension")
    print("=" * 60)
    print("Fixing: ValueError: numpy.dtype size changed, may indicate binary incompatibility")
    print()

def check_current_versions():
    """Check current package versions"""
    print("📋 Current package versions:")
    
    packages = ['numpy', 'scipy', 'torch', 'torchvision']
    versions = {}
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            versions[package] = version
            print(f"  {package}: {version}")
        except ImportError:
            versions[package] = 'Not installed'
            print(f"  {package}: Not installed")
    
    print()
    return versions

def backup_current_environment():
    """Create a backup of current pip freeze"""
    print("💾 Creating environment backup...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "freeze"
        ], capture_output=True, text=True, check=True)
        
        backup_file = Path("pip_freeze_backup.txt")
        with open(backup_file, 'w') as f:
            f.write(result.stdout)
        
        print(f"✅ Environment backed up to {backup_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create backup: {e}")
        return False

def uninstall_conflicting_packages():
    """Uninstall packages that may cause conflicts"""
    print("🗑️ Removing conflicting packages...")
    
    packages_to_remove = [
        'numpy', 'scipy', 'pandas', 'scikit-learn', 
        'matplotlib', 'seaborn', 'opencv-python'
    ]
    
    for package in packages_to_remove:
        try:
            print(f"  Removing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", "-y", package
            ], check=False, capture_output=True)
        except Exception:
            pass  # Continue even if uninstall fails
    
    print("✅ Conflicting packages removed")

def install_compatible_numpy():
    """Install numpy with specific version constraints"""
    print("📦 Installing compatible numpy...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--force-reinstall",
            "numpy>=1.24.0,<2.0.0"
        ], check=True)
        
        print("✅ Compatible numpy installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install numpy: {e}")
        return False

def install_compatible_scipy():
    """Install scipy compatible with the numpy version"""
    print("🔬 Installing compatible scipy...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--force-reinstall",
            "scipy>=1.10.0,<1.12.0"
        ], check=True)
        
        print("✅ Compatible scipy installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install scipy: {e}")
        return False

def reinstall_dependent_packages():
    """Reinstall packages that depend on numpy/scipy"""
    print("🔄 Reinstalling dependent packages...")
    
    dependent_packages = [
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0", 
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0"
    ]
    
    for package in dependent_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", package
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to install {package} (optional)")
    
    print("✅ Dependent packages reinstalled")

def verify_fix():
    """Verify that the fix worked"""
    print("🔍 Verifying fix...")
    
    try:
        # Test numpy import and basic operations
        import numpy as np
        test_array = np.array([1, 2, 3, 4, 5])
        test_result = np.mean(test_array)
        print(f"✅ Numpy {np.__version__} working (test result: {test_result})")
        
        # Test scipy import and basic operations
        import scipy
        from scipy import stats
        test_data = [1, 2, 3, 4, 5]
        test_stat = stats.describe(test_data)
        print(f"✅ SciPy {scipy.__version__} working")
        
        # Test torch compatibility
        try:
            import torch
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            print(f"✅ PyTorch {torch.__version__} compatible")
            
            if torch.cuda.is_available():
                print(f"✅ CUDA {torch.version.cuda} available")
        except ImportError:
            print("⚠️ PyTorch not installed (install separately)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def clean_pip_cache():
    """Clean pip cache to avoid cached incompatible wheels"""
    print("🧹 Cleaning pip cache...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "cache", "purge"
        ], check=True, capture_output=True)
        
        print("✅ Pip cache cleaned")
        return True
        
    except subprocess.CalledProcessError:
        print("⚠️ Could not clean pip cache (non-critical)")
        return False

def main():
    """Main fix process"""
    print_header()
    
    # Check current versions
    original_versions = check_current_versions()
    
    # Create backup
    backup_current_environment()
    
    # Clean pip cache first
    clean_pip_cache()
    
    # Remove conflicting packages
    uninstall_conflicting_packages()
    
    # Install compatible versions
    if not install_compatible_numpy():
        print("❌ Critical: Failed to install compatible numpy")
        sys.exit(1)
    
    if not install_compatible_scipy():
        print("❌ Critical: Failed to install compatible scipy")
        sys.exit(1)
    
    # Reinstall dependent packages
    reinstall_dependent_packages()
    
    # Verify the fix
    if verify_fix():
        print("\n🎉 Numpy compatibility fix completed successfully!")
        print("📖 The 'numpy.dtype size changed' error should now be resolved")
        print("🔄 Restart your Python environment/kernel if needed")
    else:
        print("\n❌ Fix verification failed")
        print("📋 Check the error messages above")
        print("🔧 You may need to manually resolve remaining issues")
    
    # Show final versions
    print("\n📋 Final package versions:")
    check_current_versions()

if __name__ == "__main__":
    main()
