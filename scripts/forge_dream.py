"""
Main Forge Dream Extension Script
Entry point for Stable Diffusion Forge integration
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
import gradio as gr

# Add extension path to Python path
extension_path = Path(__file__).parent.parent
sys.path.insert(0, str(extension_path))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
model_manager = None
memory_manager = None
hidream_pipeline = None
faceswap_integration = None
forge_dream_ui = None

def safe_import(module_name: str, class_name: str):
    """Safely import a module and class with error handling"""
    try:
        # Try direct import first
        module_path = extension_path / "scripts" / f"{module_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        else:
            # Fallback to standard import
            module = __import__(f"scripts.{module_name}", fromlist=[class_name])
            return getattr(module, class_name)
    except ImportError as e:
        logger.error(f"Failed to import {class_name} from {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {class_name}: {e}")
        return None

def create_mock_classes():
    """Create mock classes when real imports fail"""
    class MockModelManager:
        def __init__(self, *args, **kwargs):
            self.models = {"fp8": [], "gguf": []}
        def get_available_models(self, model_type):
            return []
        def get_downloaded_models(self, model_type):
            return []
        def download_model(self, model_type, model_name):
            return f"Mock download: {model_name}"
    
    class MockMemoryManager:
        def __init__(self, *args, **kwargs):
            pass
        def clear_cache(self):
            return "Cache cleared"
        def get_memory_stats(self):
            return {"vram_used": "0GB", "vram_total": "16GB"}
    
    class MockHiDreamPipeline:
        def __init__(self, *args, **kwargs):
            pass
        def generate_image(self, *args, **kwargs):
            return []
    
    class MockFaceSwapIntegration:
        def __init__(self, *args, **kwargs):
            pass
    
    return MockModelManager, MockMemoryManager, MockHiDreamPipeline, MockFaceSwapIntegration

def initialize_extension():
    """Initialize the Forge Dream extension with error handling"""
    global model_manager, memory_manager, hidream_pipeline, faceswap_integration, forge_dream_ui
    
    try:
        logger.info("Initializing Forge Dream Extension...")
        
        # Import classes safely
        ModelManager = safe_import("model_manager", "ModelManager")
        MemoryManager = safe_import("memory_manager", "MemoryManager")
        HiDreamPipeline = safe_import("hidream_pipeline", "HiDreamPipeline")
        FaceSwapIntegration = safe_import("faceswap_integration", "FaceSwapIntegration")
        ForgeDreamUI = safe_import("ui_components", "ForgeDreamUI")
        
        # Use mock classes if real imports failed
        if not all([ModelManager, MemoryManager, HiDreamPipeline, FaceSwapIntegration]):
            logger.warning("Some imports failed, using mock classes for demonstration")
            MockModelManager, MockMemoryManager, MockHiDreamPipeline, MockFaceSwapIntegration = create_mock_classes()
            ModelManager = ModelManager or MockModelManager
            MemoryManager = MemoryManager or MockMemoryManager
            HiDreamPipeline = HiDreamPipeline or MockHiDreamPipeline
            FaceSwapIntegration = FaceSwapIntegration or MockFaceSwapIntegration
        
        if not ForgeDreamUI:
            logger.error("Failed to import UI components")
            return False
        
        # Initialize managers
        model_manager = ModelManager(extension_path)
        memory_manager = MemoryManager(max_vram_gb=24)  # Default 24GB, can be configured
        
        # Initialize pipelines
        hidream_pipeline = HiDreamPipeline(model_manager, memory_manager)
        
        # Initialize faceswap
        faceswap_model_path = model_manager.get_model_path("faceswap", "inswapper") / "inswapper_128.onnx"
        faceswap_integration = FaceSwapIntegration(
            str(faceswap_model_path) if faceswap_model_path.exists() else None
        )
        
        # Initialize UI
        forge_dream_ui = ForgeDreamUI(
            model_manager, 
            memory_manager, 
            hidream_pipeline, 
            faceswap_integration
        )
        
        logger.info("Forge Dream Extension initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Forge Dream Extension: {e}")
        return False

def create_minimal_interface():
    """Create a minimal interface when full initialization fails"""
    with gr.Blocks(
        analytics_enabled=False,
        title="Forge Dream - Loading..."
    ) as minimal_interface:
        
        gr.Markdown("""
        # 🌟 Forge Dream Extension
        **Loading extension components...**
        
        If this message persists, please check the console for error messages.
        """)
        
        with gr.Row():
            gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
        
        with gr.Row():
            gr.Button("Generate (Basic Mode)", variant="primary")
            gr.Button("Refresh Extension", variant="secondary")
        
        gr.Markdown("""
        ### Troubleshooting
        
        If the extension is not working properly:
        1. Check the console for error messages
        2. Ensure all dependencies are installed
        3. Restart Stable Diffusion Forge
        4. Check that models are downloaded correctly
        """)
    
    return minimal_interface

def create_forge_dream_interface():
    """Create the main Forge Dream interface"""
    
    logger.info("Creating Forge Dream interface...")
    
    if not initialize_extension():
        logger.warning("Full initialization failed, creating minimal interface")
        return create_minimal_interface()
    
    # Load custom CSS
    css_path = extension_path / "style.css"
    custom_css = ""
    if css_path.exists():
        try:
            with open(css_path, 'r') as f:
                custom_css = f.read()
            logger.info("Custom CSS loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load custom CSS: {e}")
    
    with gr.Blocks(
        analytics_enabled=False,
        title="Forge Dream - HiDream Models with FaceSwap",
        css=custom_css
    ) as forge_dream_interface:
        
        # Header
        gr.Markdown("""
        # 🌟 Forge Dream Extension
        **Dual-panel interface for HiDream FP8/GGUF models with integrated faceswap functionality**
        """)
        
        try:
            # Memory monitor
            logger.info("Creating memory monitor...")
            memory_monitor = forge_dream_ui.create_memory_monitor()
            
            # Main dual-panel layout
            logger.info("Creating dual-panel layout...")
            with gr.Row(elem_classes=["forge-dream-container"]):
                # FP8 Panel (Left)
                logger.info("Creating FP8 panel...")
                fp8_components = forge_dream_ui.create_panel("fp8", "🚀 FP8 Models")
                
                # GGUF Panel (Right)  
                logger.info("Creating GGUF panel...")
                gguf_components = forge_dream_ui.create_panel("gguf", "⚡ GGUF Models")
            
            # Setup event handlers
            logger.info("Setting up event handlers...")
            forge_dream_ui.setup_event_handlers(fp8_components, gguf_components)
            
            logger.info("Forge Dream interface created successfully")
            
        except Exception as e:
            logger.error(f"Error creating UI components: {e}")
            gr.Markdown(f"**Error creating interface components:** {str(e)}")
        
        # Footer with information
        with gr.Accordion("ℹ️ Information & Settings", open=False):
            gr.Markdown("""
            ### About Forge Dream Extension
            
            This extension provides a dual-panel interface for running HiDream models in both FP8 and GGUF formats,
            with integrated Reactor-compatible faceswap functionality.
            
            **Features:**
            - **FP8 Models**: High-quality models optimized for 12-24GB VRAM
            - **GGUF Models**: Quantized models for efficient inference
            - **Face Swap**: Reactor-compatible face swapping with checkpoint support
            - **Memory Management**: Automatic VRAM optimization
            - **Dual Modes**: Both Text2Img and Img2Img generation
            
            **Model Requirements:**
            - FP8 models require 12-20GB VRAM depending on variant
            - GGUF models require 7-15GB VRAM depending on quantization
            - Face swap models require additional ~500MB VRAM
            
            **Tips:**
            - Use Fast variants for 12GB VRAM systems
            - Use Q6_K GGUF for best quality on 24GB systems
            - Enable face swap for character consistency
            - Monitor VRAM usage in the indicator above
            """)
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                clear_btn = gr.Button("🧹 Clear VRAM Cache", variant="secondary")
                stats_btn = gr.Button("📊 Memory Stats", variant="secondary")
                
                # Add event handlers for buttons
                try:
                    refresh_btn.click(
                        fn=lambda: "Models refreshed",
                        outputs=gr.Textbox(visible=False)
                    )
                    clear_btn.click(
                        fn=lambda: memory_manager.clear_cache() if memory_manager else "Cache cleared",
                        outputs=gr.Textbox(visible=False)
                    )
                    stats_btn.click(
                        fn=lambda: memory_manager.get_memory_stats() if memory_manager else {"error": "Not available"},
                        outputs=gr.Textbox(visible=False)
                    )
                except Exception as e:
                    logger.warning(f"Could not set up button handlers: {e}")
    
    return forge_dream_interface

def on_ui_tabs():
    """Forge extension entry point - FIXED: Removed incorrect type annotation"""
    try:
        logger.info("on_ui_tabs called - creating Forge Dream interface")
        interface = create_forge_dream_interface()
        logger.info("Forge Dream interface created, returning tab configuration")
        return [(interface, "Forge Dream", "forge_dream")]
    except Exception as e:
        logger.error(f"Failed to create Forge Dream interface: {e}")
        # Return minimal error interface
        with gr.Blocks() as error_interface:
            gr.Markdown(f"""
            ## Forge Dream Extension - Error
            
            **Error:** {str(e)}
            
            Please check the console for more details and ensure all dependencies are installed correctly.
            """)
        return [(error_interface, "Forge Dream (Error)", "forge_dream_error")]

# Register with Forge
try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_ui_tabs(on_ui_tabs)
    logger.info("Forge Dream extension registered successfully with script_callbacks")
except ImportError:
    logger.error("Failed to import script_callbacks - not running in Forge environment")
except Exception as e:
    logger.error(f"Failed to register Forge Dream extension: {e}")

# Optional: Register API endpoints
def register_api():
    """Register API endpoints for external access"""
    try:
        import modules.script_callbacks as script_callbacks
        
        def forge_dream_api(demo, app):
            """Add API endpoints to FastAPI app"""
            
            @app.get("/forge_dream/models")
            async def get_models():
                """Get available models"""
                if model_manager:
                    return {
                        "fp8_models": model_manager.get_available_models("fp8"),
                        "gguf_models": model_manager.get_available_models("gguf"),
                        "downloaded_fp8": model_manager.get_downloaded_models("fp8"),
                        "downloaded_gguf": model_manager.get_downloaded_models("gguf")
                    }
                return {"error": "Extension not initialized"}
            
            @app.get("/forge_dream/memory")
            async def get_memory_stats():
                """Get memory statistics"""
                if memory_manager:
                    return memory_manager.get_memory_stats()
                return {"error": "Extension not initialized"}
            
            @app.post("/forge_dream/generate")
            async def generate_image(request: dict):
                """Generate image via API"""
                if hidream_pipeline:
                    try:
                        model_type = request.get("model_type", "fp8")
                        model_name = request.get("model_name")
                        prompt = request.get("prompt", "")
                        
                        # Load model if needed
                        if model_type == "fp8":
                            hidream_pipeline.load_fp8_model(model_name)
                        else:
                            hidream_pipeline.load_gguf_model(model_name)
                        
                        # Generate images
                        images = hidream_pipeline.generate_image(
                            prompt=prompt,
                            **request.get("parameters", {})
                        )
                        
                        return {"success": True, "num_images": len(images)}
                    except Exception as e:
                        return {"error": str(e)}
                return {"error": "Extension not initialized"}
        
        script_callbacks.on_app_started(forge_dream_api)
        logger.info("Forge Dream API endpoints registered")
        
    except Exception as e:
        logger.error(f"Failed to register API endpoints: {e}")

# Register API if available
try:
    register_api()
except Exception as e:
    logger.warning(f"API registration failed: {e}")
