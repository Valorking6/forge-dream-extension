"""
Enhanced main module for Forge Dream extension with backend integration.
This is the main entry point that integrates with Forge WebUI.
"""

import os
import sys
import gradio as gr
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our fixed components
try:
    from backend import backend, ForgeDreamBackend
    from ui_components_fixed import main_interface, advanced_interface
    print("✅ Successfully imported backend and UI components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback to basic implementation
    backend = None
    main_interface = None
    advanced_interface = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Extension metadata
EXTENSION_NAME = "Forge Dream"
EXTENSION_VERSION = "1.2.0"
EXTENSION_DESCRIPTION = "Advanced AI image generation with comprehensive backend support"

class ForgeDreamExtension:
    """Main extension class for Forge Dream."""
    
    def __init__(self):
        self.backend = backend
        self.is_initialized = False
        self.extension_dir = current_dir
        
    def initialize(self):
        """Initialize the extension."""
        try:
            logger.info(f"Initializing {EXTENSION_NAME} v{EXTENSION_VERSION}")
            
            # Ensure backend is available
            if self.backend is None:
                logger.warning("Backend not available, creating fallback")
                self.backend = self._create_fallback_backend()
            
            # Initialize backend
            if hasattr(self.backend, 'initialize'):
                self.backend.initialize()
            
            self.is_initialized = True
            logger.info("✅ Forge Dream extension initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize extension: {e}")
            raise
    
    def _create_fallback_backend(self):
        """Create a fallback backend if main backend is not available."""
        class FallbackBackend:
            def __init__(self):
                self.current_settings = {
                    'model': 'stable-diffusion-v1-5',
                    'steps': 20,
                    'cfg_scale': 7.5,
                    'width': 512,
                    'height': 512,
                    'seed': -1
                }
            
            def generate_image(self, prompt, negative_prompt="", **kwargs):
                import numpy as np
                # Return a simple placeholder image
                img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                info = {'prompt': prompt, 'status': 'fallback_generation'}
                return [img], info
        
        return FallbackBackend()
    
    def create_ui(self):
        """Create the main UI interface."""
        try:
            if not self.is_initialized:
                self.initialize()
            
            # Use the fixed interface if available
            if main_interface is not None:
                logger.info("Using main interface with backend integration")
                return main_interface
            else:
                logger.warning("Main interface not available, creating fallback UI")
                return self._create_fallback_ui()
                
        except Exception as e:
            logger.error(f"Error creating UI: {e}")
            return self._create_fallback_ui()
    
    def create_advanced_ui(self):
        """Create the advanced UI interface."""
        try:
            if not self.is_initialized:
                self.initialize()
            
            if advanced_interface is not None:
                logger.info("Using advanced interface")
                return advanced_interface
            else:
                return self.create_ui()  # Fallback to main UI
                
        except Exception as e:
            logger.error(f"Error creating advanced UI: {e}")
            return self.create_ui()
    
    def _create_fallback_ui(self):
        """Create a simple fallback UI."""
        with gr.Blocks(title="Forge Dream (Fallback)") as fallback_ui:
            gr.Markdown("# 🎨 Forge Dream - Fallback Mode")
            gr.Markdown("⚠️ Running in fallback mode. Some features may be limited.")
            
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt...")
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output = gr.Image(label="Generated Image")
                    status = gr.Textbox(label="Status", interactive=False)
            
            def fallback_generate(prompt_text):
                try:
                    if self.backend:
                        images, info = self.backend.generate_image(prompt_text)
                        return images[0] if images else None, "Generated successfully"
                    else:
                        return None, "Backend not available"
                except Exception as e:
                    return None, f"Error: {str(e)}"
            
            generate_btn.click(
                fn=fallback_generate,
                inputs=[prompt],
                outputs=[output, status]
            )
        
        return fallback_ui
    
    def get_extension_info(self):
        """Get extension information."""
        return {
            'name': EXTENSION_NAME,
            'version': EXTENSION_VERSION,
            'description': EXTENSION_DESCRIPTION,
            'initialized': self.is_initialized,
            'backend_available': self.backend is not None,
            'ui_available': main_interface is not None
        }

# Global extension instance
forge_dream_extension = ForgeDreamExtension()

# Forge WebUI integration functions
def on_ui_tabs():
    """Called by Forge WebUI to create tabs."""
    try:
        logger.info("Creating Forge Dream UI tabs")
        
        # Create main tab
        main_ui = forge_dream_extension.create_ui()
        main_tab = (main_ui, "Forge Dream", "forge_dream_main")
        
        # Create advanced tab
        advanced_ui = forge_dream_extension.create_advanced_ui()
        advanced_tab = (advanced_ui, "Forge Dream Advanced", "forge_dream_advanced")
        
        return [main_tab, advanced_tab]
        
    except Exception as e:
        logger.error(f"Error creating UI tabs: {e}")
        # Return a simple error tab
        with gr.Blocks() as error_ui:
            gr.Markdown(f"# ❌ Forge Dream Error\n\nFailed to load extension: {str(e)}")
        
        return [(error_ui, "Forge Dream (Error)", "forge_dream_error")]

def on_ui_settings():
    """Called by Forge WebUI to add settings."""
    try:
        # Add extension settings to Forge WebUI settings
        settings = [
            {
                'key': 'forge_dream_enabled',
                'label': 'Enable Forge Dream Extension',
                'type': 'checkbox',
                'default': True,
                'section': 'forge_dream'
            },
            {
                'key': 'forge_dream_default_model',
                'label': 'Default Model',
                'type': 'dropdown',
                'choices': ['stable-diffusion-v1-5', 'stable-diffusion-v2-1', 'stable-diffusion-xl-base-1.0'],
                'default': 'stable-diffusion-v1-5',
                'section': 'forge_dream'
            },
            {
                'key': 'forge_dream_max_batch_size',
                'label': 'Maximum Batch Size',
                'type': 'slider',
                'minimum': 1,
                'maximum': 16,
                'default': 4,
                'section': 'forge_dream'
            }
        ]
        
        return settings
        
    except Exception as e:
        logger.error(f"Error adding settings: {e}")
        return []

def script_callbacks():
    """Called by Forge WebUI for script callbacks."""
    try:
        logger.info("Setting up Forge Dream script callbacks")
        
        # Initialize extension
        forge_dream_extension.initialize()
        
        # Log extension info
        info = forge_dream_extension.get_extension_info()
        logger.info(f"Extension info: {info}")
        
    except Exception as e:
        logger.error(f"Error in script callbacks: {e}")

# Entry point for testing
def main():
    """Main entry point for standalone testing."""
    print(f"🚀 Starting {EXTENSION_NAME} v{EXTENSION_VERSION}")
    
    try:
        # Initialize extension
        forge_dream_extension.initialize()
        
        # Create and launch UI
        ui = forge_dream_extension.create_ui()
        
        print("✅ Launching Forge Dream interface...")
        ui.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        print(f"❌ Error launching extension: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Export for Forge WebUI
__all__ = [
    'forge_dream_extension',
    'on_ui_tabs',
    'on_ui_settings', 
    'script_callbacks',
    'ForgeDreamExtension'
]