"""
Complete backend implementation for Forge Dream extension.
Provides proper handler functions for all interactive components to resolve
"IndexError: function has no backend method" errors.
"""

import gradio as gr
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForgeDreamBackend:
    """Main backend class handling all Forge Dream functionality."""
    
    def __init__(self):
        self.current_settings = {
            'model': 'stable-diffusion-v1-5',
            'steps': 20,
            'cfg_scale': 7.5,
            'width': 512,
            'height': 512,
            'seed': -1,
            'sampler': 'DPM++ 2M Karras',
            'batch_size': 1,
            'batch_count': 1
        }
        self.generation_history = []
        self.available_models = [
            'stable-diffusion-v1-5',
            'stable-diffusion-v2-1',
            'stable-diffusion-xl-base-1.0',
            'dreamshaper-8',
            'realistic-vision-v5.1'
        ]
        self.available_samplers = [
            'Euler a',
            'Euler',
            'LMS',
            'Heun',
            'DPM2',
            'DPM2 a',
            'DPM++ 2S a',
            'DPM++ 2M',
            'DPM++ SDE',
            'DPM++ 2M Karras',
            'DPM++ SDE Karras',
            'DDIM',
            'PLMS'
        ]
        
    def generate_image(self, prompt: str, negative_prompt: str = "", **kwargs) -> Tuple[List[np.ndarray], Dict]:
        """
        Generate images based on prompt and settings.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt to avoid certain elements
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_images, generation_info)
        """
        try:
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            
            # Update settings with provided kwargs
            settings = self.current_settings.copy()
            settings.update(kwargs)
            
            # Simulate image generation (replace with actual implementation)
            batch_size = settings.get('batch_size', 1)
            width = settings.get('width', 512)
            height = settings.get('height', 512)
            
            # Generate placeholder images (replace with actual model inference)
            images = []
            for i in range(batch_size):
                # Create a simple gradient image as placeholder
                img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                images.append(img)
            
            # Generation info
            gen_info = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'settings': settings,
                'seed_used': settings.get('seed', -1),
                'steps': settings.get('steps', 20),
                'cfg_scale': settings.get('cfg_scale', 7.5),
                'sampler': settings.get('sampler', 'DPM++ 2M Karras'),
                'model': settings.get('model', 'stable-diffusion-v1-5')
            }
            
            # Add to history
            self.generation_history.append(gen_info)
            
            logger.info(f"Successfully generated {len(images)} image(s)")
            return images, gen_info
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    def update_model(self, model_name: str) -> str:
        """Update the current model."""
        if model_name in self.available_models:
            self.current_settings['model'] = model_name
            logger.info(f"Model updated to: {model_name}")
            return f"Model changed to: {model_name}"
        else:
            return f"Model {model_name} not available"
    
    def update_sampler(self, sampler_name: str) -> str:
        """Update the current sampler."""
        if sampler_name in self.available_samplers:
            self.current_settings['sampler'] = sampler_name
            logger.info(f"Sampler updated to: {sampler_name}")
            return f"Sampler changed to: {sampler_name}"
        else:
            return f"Sampler {sampler_name} not available"
    
    def update_steps(self, steps: int) -> str:
        """Update sampling steps."""
        steps = max(1, min(150, int(steps)))
        self.current_settings['steps'] = steps
        return f"Steps set to: {steps}"
    
    def update_cfg_scale(self, cfg_scale: float) -> str:
        """Update CFG scale."""
        cfg_scale = max(1.0, min(30.0, float(cfg_scale)))
        self.current_settings['cfg_scale'] = cfg_scale
        return f"CFG Scale set to: {cfg_scale}"
    
    def update_dimensions(self, width: int, height: int) -> str:
        """Update image dimensions."""
        width = max(64, min(2048, int(width)))
        height = max(64, min(2048, int(height)))
        self.current_settings['width'] = width
        self.current_settings['height'] = height
        return f"Dimensions set to: {width}x{height}"
    
    def update_seed(self, seed: int) -> str:
        """Update generation seed."""
        self.current_settings['seed'] = int(seed)
        return f"Seed set to: {seed}"
    
    def randomize_seed(self) -> Tuple[int, str]:
        """Generate a random seed."""
        import random
        new_seed = random.randint(0, 2**32 - 1)
        self.current_settings['seed'] = new_seed
        return new_seed, f"Seed randomized to: {new_seed}"
    
    def update_batch_settings(self, batch_size: int, batch_count: int) -> str:
        """Update batch generation settings."""
        batch_size = max(1, min(8, int(batch_size)))
        batch_count = max(1, min(10, int(batch_count)))
        self.current_settings['batch_size'] = batch_size
        self.current_settings['batch_count'] = batch_count
        return f"Batch settings: {batch_size} images per batch, {batch_count} batches"
    
    def get_generation_history(self) -> List[Dict]:
        """Get the generation history."""
        return self.generation_history[-10:]  # Return last 10 generations
    
    def clear_history(self) -> str:
        """Clear generation history."""
        self.generation_history.clear()
        return "Generation history cleared"
    
    def save_settings(self, filename: str = "forge_dream_settings.json") -> str:
        """Save current settings to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_settings, f, indent=2)
            return f"Settings saved to {filename}"
        except Exception as e:
            return f"Error saving settings: {str(e)}"
    
    def load_settings(self, filename: str = "forge_dream_settings.json") -> str:
        """Load settings from file."""
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    loaded_settings = json.load(f)
                self.current_settings.update(loaded_settings)
                return f"Settings loaded from {filename}"
            else:
                return f"Settings file {filename} not found"
        except Exception as e:
            return f"Error loading settings: {str(e)}"
    
    def get_current_settings(self) -> Dict:
        """Get current settings as dictionary."""
        return self.current_settings.copy()
    
    def process_image_upload(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process uploaded image for img2img generation."""
        if image is None:
            return None, "No image uploaded"
        
        logger.info(f"Processing uploaded image with shape: {image.shape}")
        return image, f"Image processed: {image.shape[1]}x{image.shape[0]}"
    
    def apply_style_preset(self, style_name: str) -> str:
        """Apply predefined style presets."""
        style_presets = {
            'photorealistic': {
                'cfg_scale': 7.0,
                'steps': 30,
                'sampler': 'DPM++ 2M Karras'
            },
            'artistic': {
                'cfg_scale': 12.0,
                'steps': 25,
                'sampler': 'Euler a'
            },
            'anime': {
                'cfg_scale': 8.0,
                'steps': 28,
                'sampler': 'DPM++ SDE Karras'
            },
            'fantasy': {
                'cfg_scale': 10.0,
                'steps': 35,
                'sampler': 'DDIM'
            }
        }
        
        if style_name in style_presets:
            self.current_settings.update(style_presets[style_name])
            return f"Applied {style_name} style preset"
        else:
            return f"Style preset {style_name} not found"

# Global backend instance
backend = ForgeDreamBackend()

# Handler functions for Gradio components
def handle_generate_click(prompt, negative_prompt, steps, cfg_scale, width, height, seed, sampler, model, batch_size):
    """Handler for generate button click."""
    try:
        # Update settings
        backend.update_steps(steps)
        backend.update_cfg_scale(cfg_scale)
        backend.update_dimensions(width, height)
        backend.update_seed(seed)
        backend.update_sampler(sampler)
        backend.update_model(model)
        backend.current_settings['batch_size'] = batch_size
        
        # Generate images
        images, gen_info = backend.generate_image(prompt, negative_prompt)
        
        # Return first image and generation info
        if images:
            return images[0], json.dumps(gen_info, indent=2)
        else:
            return None, "No images generated"
            
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return None, f"Error: {str(e)}"

def handle_model_change(model_name):
    """Handler for model dropdown change."""
    return backend.update_model(model_name)

def handle_sampler_change(sampler_name):
    """Handler for sampler dropdown change."""
    return backend.update_sampler(sampler_name)

def handle_randomize_seed():
    """Handler for randomize seed button."""
    new_seed, message = backend.randomize_seed()
    return new_seed, message

def handle_style_preset_change(style_name):
    """Handler for style preset dropdown."""
    message = backend.apply_style_preset(style_name)
    settings = backend.get_current_settings()
    return (
        settings['cfg_scale'],
        settings['steps'], 
        settings['sampler'],
        message
    )

def handle_save_settings():
    """Handler for save settings button."""
    return backend.save_settings()

def handle_load_settings():
    """Handler for load settings button."""
    message = backend.load_settings()
    settings = backend.get_current_settings()
    return (
        settings['steps'],
        settings['cfg_scale'],
        settings['width'],
        settings['height'],
        settings['seed'],
        settings['sampler'],
        settings['model'],
        message
    )

def handle_clear_history():
    """Handler for clear history button."""
    return backend.clear_history()

def handle_image_upload(image):
    """Handler for image upload."""
    return backend.process_image_upload(image)

def handle_batch_settings_change(batch_size, batch_count):
    """Handler for batch settings change."""
    return backend.update_batch_settings(batch_size, batch_count)

# Export all handler functions for use in UI
__all__ = [
    'ForgeDreamBackend',
    'backend',
    'handle_generate_click',
    'handle_model_change',
    'handle_sampler_change',
    'handle_randomize_seed',
    'handle_style_preset_change',
    'handle_save_settings',
    'handle_load_settings',
    'handle_clear_history',
    'handle_image_upload',
    'handle_batch_settings_change'
]