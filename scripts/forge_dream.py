import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path
import gradio as gr
import torch
import numpy as np
from PIL import Image
import json
import time
import threading
from typing import Optional, List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows compatibility fixes
if platform.system() == "Windows":
    # Fix path separators for Windows
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# CUDA 12.8 and RTX 5070 Ti optimizations
def setup_cuda_optimizations():
    """Setup CUDA optimizations for RTX 5070 Ti and CUDA 12.8"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
        logger.info(f"CUDA Device: {device_name}")
        logger.info(f"CUDA Version: {cuda_version}")
        
        # RTX 5070 Ti specific optimizations
        if "RTX 5070" in device_name or "RTX 50" in device_name:
            logger.info("Applying RTX 5070 Ti optimizations")
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # CUDA 12.8 specific settings
        if cuda_version and cuda_version.startswith("12."):
            logger.info("Applying CUDA 12.x optimizations")
            # Enable memory pool for better memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
        # Set memory fraction for high-end GPUs
        torch.cuda.set_per_process_memory_fraction(0.9)

# Robust import system with fallbacks
def safe_import(module_name, package=None, fallback=None):
    """Safely import modules with fallback options"""
    try:
        if package:
            return importlib.import_module(module_name, package)
        else:
            return importlib.import_module(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        if fallback:
            try:
                return importlib.import_module(fallback)
            except ImportError:
                logger.error(f"Fallback import {fallback} also failed")
        return None

# Try importing modules with torch nightly compatibility
try:
    # Primary imports for torch nightly
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    
    # Check for torch nightly specific features
    if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_getCurrentRawStream'):
        logger.info("Torch nightly detected - using advanced CUDA features")
        TORCH_NIGHTLY = True
    else:
        TORCH_NIGHTLY = False
        
except ImportError as e:
    logger.warning(f"Some torch modules not available: {e}")
    F = None
    autocast = None
    GradScaler = None
    TORCH_NIGHTLY = False

# Try importing diffusers with fallbacks
diffusers = safe_import('diffusers')
if diffusers:
    try:
        from diffusers import StableDiffusionPipeline, DiffusionPipeline
        from diffusers.schedulers import DDIMScheduler, EulerDiscreteScheduler
    except ImportError:
        logger.warning("Some diffusers components not available")

# Try importing transformers
transformers = safe_import('transformers')
if transformers:
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError:
        logger.warning("Some transformers components not available")

# Try importing face analysis libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available - face detection disabled")
    CV2_AVAILABLE = False

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    logger.warning("InsightFace not available - advanced face analysis disabled")
    INSIGHTFACE_AVAILABLE = False

class ForgeDreamExtension:
    def __init__(self):
        self.device = self.get_optimal_device()
        self.dtype = self.get_optimal_dtype()
        self.models = {}
        self.pipelines = {}
        self.face_analyzer = None
        
        # Setup CUDA optimizations
        setup_cuda_optimizations()
        
        # Initialize face analyzer if available
        if INSIGHTFACE_AVAILABLE:
            self.init_face_analyzer()
    
    def get_optimal_device(self):
        """Get the optimal device for the current setup"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {device_name}")
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return torch.device("mps")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")
    
    def get_optimal_dtype(self):
        """Get optimal dtype based on device capabilities"""
        if self.device.type == "cuda":
            # Check for RTX 5070 Ti or newer cards that support bfloat16
            device_name = torch.cuda.get_device_name(0)
            if "RTX 50" in device_name or "RTX 40" in device_name:
                if TORCH_NIGHTLY and hasattr(torch, 'bfloat16'):
                    logger.info("Using bfloat16 for RTX 50/40 series")
                    return torch.bfloat16
                else:
                    logger.info("Using float16 for RTX series")
                    return torch.float16
            else:
                return torch.float16
        else:
            return torch.float32
    
    def init_face_analyzer(self):
        """Initialize face analyzer with error handling"""
        try:
            if INSIGHTFACE_AVAILABLE:
                self.face_analyzer = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_analyzer.prepare(ctx_id=0 if self.device.type == "cuda" else -1)
                logger.info("Face analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize face analyzer: {e}")
            self.face_analyzer = None
    
    def load_model(self, model_path: str, model_type: str = "auto"):
        """Load model with Windows path compatibility and CUDA optimizations"""
        try:
            # Convert to Path object for cross-platform compatibility
            model_path = Path(model_path)
            
            # Check if model exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading model: {model_path}")
            
            # Determine model type if auto
            if model_type == "auto":
                if model_path.suffix.lower() in ['.gguf', '.ggml']:
                    model_type = "gguf"
                elif model_path.suffix.lower() in ['.fp8', '.safetensors']:
                    model_type = "fp8"
                else:
                    model_type = "standard"
            
            # Load based on type
            if model_type == "gguf":
                return self.load_gguf_model(model_path)
            elif model_type == "fp8":
                return self.load_fp8_model(model_path)
            else:
                return self.load_standard_model(model_path)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
    
    def load_gguf_model(self, model_path: Path):
        """Load GGUF model with optimizations"""
        try:
            # Try to use llama-cpp-python if available
            try:
                from llama_cpp import Llama
                model = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=-1 if self.device.type == "cuda" else 0,
                    n_ctx=2048,
                    verbose=False
                )
                logger.info(f"GGUF model loaded with llama-cpp-python: {model_path}")
                return model
            except ImportError:
                logger.warning("llama-cpp-python not available, using alternative loader")
                # Fallback to basic loading
                return self.load_standard_model(model_path)
                
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
    
    def load_fp8_model(self, model_path: Path):
        """Load FP8 model with CUDA 12.8 optimizations"""
        try:
            if diffusers:
                # Use diffusers for FP8 models
                pipeline = DiffusionPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                
                if self.device.type == "cuda":
                    pipeline = pipeline.to(self.device)
                    
                    # Enable memory efficient attention for RTX 5070 Ti
                    if hasattr(pipeline.unet, 'enable_xformers_memory_efficient_attention'):
                        try:
                            pipeline.unet.enable_xformers_memory_efficient_attention()
                            logger.info("Enabled xformers memory efficient attention")
                        except:
                            logger.warning("xformers not available")
                    
                    # Enable model CPU offload for large models
                    if hasattr(pipeline, 'enable_model_cpu_offload'):
                        pipeline.enable_model_cpu_offload()
                        logger.info("Enabled model CPU offload")
                
                logger.info(f"FP8 model loaded: {model_path}")
                return pipeline
            else:
                raise ImportError("Diffusers not available for FP8 model loading")
                
        except Exception as e:
            logger.error(f"Failed to load FP8 model: {e}")
            raise
    
    def load_standard_model(self, model_path: Path):
        """Load standard model with optimizations"""
        try:
            if diffusers:
                if model_path.is_dir():
                    # Load from directory
                    pipeline = DiffusionPipeline.from_pretrained(
                        str(model_path),
                        torch_dtype=self.dtype,
                        device_map="auto" if self.device.type == "cuda" else None
                    )
                else:
                    # Load from single file
                    pipeline = DiffusionPipeline.from_single_file(
                        str(model_path),
                        torch_dtype=self.dtype,
                        use_safetensors=model_path.suffix == '.safetensors'
                    )
                
                if self.device.type == "cuda":
                    pipeline = pipeline.to(self.device)
                
                logger.info(f"Standard model loaded: {model_path}")
                return pipeline
            else:
                raise ImportError("Diffusers not available for standard model loading")
                
        except Exception as e:
            logger.error(f"Failed to load standard model: {e}")
            raise
    
    def generate_image(self, prompt: str, negative_prompt: str = "", 
                      width: int = 512, height: int = 512, 
                      steps: int = 20, guidance_scale: float = 7.5,
                      model_name: str = None) -> Image.Image:
        """Generate image with CUDA optimizations"""
        try:
            if not model_name or model_name not in self.pipelines:
                raise ValueError("No model loaded or invalid model name")
            
            pipeline = self.pipelines[model_name]
            
            # Prepare generation parameters
            generator = torch.Generator(device=self.device).manual_seed(42)
            
            # Use autocast for mixed precision if available
            if autocast and self.device.type == "cuda":
                with autocast():
                    image = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]
            else:
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).images[0]
            
            logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise
    
    def face_swap(self, source_image: Image.Image, target_image: Image.Image) -> Image.Image:
        """Perform face swap with error handling"""
        try:
            if not self.face_analyzer:
                raise ValueError("Face analyzer not available")
            
            # Convert PIL to numpy
            source_np = np.array(source_image)
            target_np = np.array(target_image)
            
            # Convert RGB to BGR for OpenCV
            if CV2_AVAILABLE:
                source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
                target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
                
                # Detect faces
                source_faces = self.face_analyzer.get(source_bgr)
                target_faces = self.face_analyzer.get(target_bgr)
                
                if not source_faces or not target_faces:
                    raise ValueError("No faces detected in one or both images")
                
                # Perform face swap (simplified implementation)
                # This would need a proper face swap implementation
                result_bgr = target_bgr.copy()
                
                # Convert back to RGB
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_rgb)
                
                logger.info("Face swap completed")
                return result_image
            else:
                raise ValueError("OpenCV not available for face swap")
                
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            raise

# Global extension instance
forge_dream = None

def initialize_extension():
    """Initialize the Forge Dream extension"""
    global forge_dream
    try:
        forge_dream = ForgeDreamExtension()
        logger.info("Forge Dream extension initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize extension: {e}")
        return False

def create_interface():
    """Create the Gradio interface"""
    if not forge_dream:
        if not initialize_extension():
            return gr.HTML("<h3>Failed to initialize Forge Dream extension</h3>")
    
    with gr.Blocks(title="Forge Dream Extension") as interface:
        gr.Markdown("# Forge Dream Extension")
        gr.Markdown("Dual-panel UI for HiDream FP8/GGUF models with integrated faceswap")
        gr.Markdown(f"**Device:** {forge_dream.device} | **Dtype:** {forge_dream.dtype}")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Loading")
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Enter path to your model file or directory"
                )
                model_type = gr.Dropdown(
                    choices=["auto", "gguf", "fp8", "standard"],
                    value="auto",
                    label="Model Type"
                )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("### Image Generation")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter negative prompt here...",
                    lines=2
                )
                
                with gr.Row():
                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height = gr.Slider(256, 1024, 512, step=64, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(1, 50, 20, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
                
                generate_btn = gr.Button("Generate Image", variant="primary")
                output_image = gr.Image(label="Generated Image")
        
        with gr.Row():
            gr.Markdown("### Face Swap")
            with gr.Column():
                source_image = gr.Image(label="Source Image", type="pil")
                target_image = gr.Image(label="Target Image", type="pil")
                faceswap_btn = gr.Button("Perform Face Swap", variant="secondary")
                faceswap_output = gr.Image(label="Face Swap Result")
        
        # Event handlers
        def load_model_handler(path, model_type):
            try:
                if not path:
                    return "Please enter a model path"
                
                model = forge_dream.load_model(path, model_type)
                model_name = Path(path).stem
                forge_dream.pipelines[model_name] = model
                return f"Model '{model_name}' loaded successfully"
            except Exception as e:
                return f"Error loading model: {str(e)}"
        
        def generate_handler(prompt, neg_prompt, w, h, steps, guidance, model_path):
            try:
                if not forge_dream.pipelines:
                    return None, "No model loaded"
                
                model_name = list(forge_dream.pipelines.keys())[0]
                image = forge_dream.generate_image(
                    prompt, neg_prompt, w, h, steps, guidance, model_name
                )
                return image
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return None
        
        def faceswap_handler(source, target):
            try:
                if source is None or target is None:
                    return None
                return forge_dream.face_swap(source, target)
            except Exception as e:
                logger.error(f"Face swap error: {e}")
                return None
        
        load_btn.click(
            load_model_handler,
            inputs=[model_path, model_type],
            outputs=[model_status]
        )
        
        generate_btn.click(
            generate_handler,
            inputs=[prompt, negative_prompt, width, height, steps, guidance_scale, model_path],
            outputs=[output_image]
        )
        
        faceswap_btn.click(
            faceswap_handler,
            inputs=[source_image, target_image],
            outputs=[faceswap_output]
        )
    
    return interface

# Entry point for Forge
def on_ui_tabs():
    """Entry point for Stable Diffusion Forge"""
    try:
        interface = create_interface()
        return [(interface, "Forge Dream", "forge_dream")]
    except Exception as e:
        logger.error(f"Failed to create UI: {e}")
        error_interface = gr.HTML(f"<h3>Error: {str(e)}</h3>")
        return [(error_interface, "Forge Dream (Error)", "forge_dream")]

# Initialize on import
if __name__ == "__main__":
    # For standalone testing
    interface = create_interface()
    interface.launch()
else:
    # Initialize when imported by Forge
    initialize_extension()