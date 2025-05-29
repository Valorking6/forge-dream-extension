"""
Unit tests for Forge Dream backend functionality.
Tests all handler functions and backend methods to ensure proper operation.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from backend import (
        ForgeDreamBackend,
        backend,
        handle_generate_click,
        handle_model_change,
        handle_sampler_change,
        handle_randomize_seed,
        handle_style_preset_change,
        handle_save_settings,
        handle_load_settings,
        handle_clear_history,
        handle_image_upload,
        handle_batch_settings_change
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import backend: {e}")
    BACKEND_AVAILABLE = False

class TestForgeDreamBackend(unittest.TestCase):
    """Test cases for ForgeDreamBackend class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not BACKEND_AVAILABLE:
            self.skipTest("Backend not available")
        
        self.backend = ForgeDreamBackend()
    
    def test_backend_initialization(self):
        """Test backend initializes correctly."""
        self.assertIsInstance(self.backend, ForgeDreamBackend)
        self.assertIn('model', self.backend.current_settings)
        self.assertIn('steps', self.backend.current_settings)
        self.assertIn('cfg_scale', self.backend.current_settings)
        self.assertTrue(len(self.backend.available_models) > 0)
        self.assertTrue(len(self.backend.available_samplers) > 0)
    
    def test_generate_image(self):
        """Test image generation functionality."""
        prompt = "A beautiful landscape"
        images, info = self.backend.generate_image(prompt)
        
        self.assertIsInstance(images, list)
        self.assertTrue(len(images) > 0)
        self.assertIsInstance(images[0], np.ndarray)
        self.assertEqual(len(images[0].shape), 3)  # Height, Width, Channels
        self.assertIsInstance(info, dict)
        self.assertIn('prompt', info)
        self.assertEqual(info['prompt'], prompt)
    
    def test_update_model(self):
        """Test model updating."""
        original_model = self.backend.current_settings['model']
        
        # Test valid model
        if len(self.backend.available_models) > 1:
            new_model = [m for m in self.backend.available_models if m != original_model][0]
            result = self.backend.update_model(new_model)
            self.assertIn("changed to", result)
            self.assertEqual(self.backend.current_settings['model'], new_model)
        
        # Test invalid model
        result = self.backend.update_model("invalid_model")
        self.assertIn("not available", result)
    
    def test_update_sampler(self):
        """Test sampler updating."""
        original_sampler = self.backend.current_settings['sampler']
        
        # Test valid sampler
        if len(self.backend.available_samplers) > 1:
            new_sampler = [s for s in self.backend.available_samplers if s != original_sampler][0]
            result = self.backend.update_sampler(new_sampler)
            self.assertIn("changed to", result)
            self.assertEqual(self.backend.current_settings['sampler'], new_sampler)
        
        # Test invalid sampler
        result = self.backend.update_sampler("invalid_sampler")
        self.assertIn("not available", result)
    
    def test_update_steps(self):
        """Test steps updating with bounds checking."""
        # Test normal value
        result = self.backend.update_steps(25)
        self.assertEqual(self.backend.current_settings['steps'], 25)
        self.assertIn("Steps set to: 25", result)
        
        # Test lower bound
        result = self.backend.update_steps(-5)
        self.assertEqual(self.backend.current_settings['steps'], 1)
        
        # Test upper bound
        result = self.backend.update_steps(200)
        self.assertEqual(self.backend.current_settings['steps'], 150)
    
    def test_update_cfg_scale(self):
        """Test CFG scale updating with bounds checking."""
        # Test normal value
        result = self.backend.update_cfg_scale(8.5)
        self.assertEqual(self.backend.current_settings['cfg_scale'], 8.5)
        
        # Test lower bound
        result = self.backend.update_cfg_scale(0.5)
        self.assertEqual(self.backend.current_settings['cfg_scale'], 1.0)
        
        # Test upper bound
        result = self.backend.update_cfg_scale(50.0)
        self.assertEqual(self.backend.current_settings['cfg_scale'], 30.0)
    
    def test_update_dimensions(self):
        """Test dimension updating with bounds checking."""
        # Test normal values
        result = self.backend.update_dimensions(768, 768)
        self.assertEqual(self.backend.current_settings['width'], 768)
        self.assertEqual(self.backend.current_settings['height'], 768)
        self.assertIn("768x768", result)
        
        # Test bounds
        result = self.backend.update_dimensions(32, 3000)
        self.assertEqual(self.backend.current_settings['width'], 64)  # Minimum
        self.assertEqual(self.backend.current_settings['height'], 2048)  # Maximum
    
    def test_randomize_seed(self):
        """Test seed randomization."""
        original_seed = self.backend.current_settings['seed']
        new_seed, message = self.backend.randomize_seed()
        
        self.assertIsInstance(new_seed, int)
        self.assertNotEqual(new_seed, original_seed)
        self.assertEqual(self.backend.current_settings['seed'], new_seed)
        self.assertIn("randomized", message.lower())
    
    def test_batch_settings(self):
        """Test batch settings updating."""
        result = self.backend.update_batch_settings(4, 2)
        self.assertEqual(self.backend.current_settings['batch_size'], 4)
        self.assertEqual(self.backend.current_settings['batch_count'], 2)
        self.assertIn("4 images per batch", result)
        self.assertIn("2 batches", result)
        
        # Test bounds
        result = self.backend.update_batch_settings(20, 50)
        self.assertEqual(self.backend.current_settings['batch_size'], 8)  # Max
        self.assertEqual(self.backend.current_settings['batch_count'], 10)  # Max
    
    def test_generation_history(self):
        """Test generation history functionality."""
        # Clear history first
        self.backend.clear_history()
        self.assertEqual(len(self.backend.get_generation_history()), 0)
        
        # Generate an image to add to history
        self.backend.generate_image("test prompt")
        history = self.backend.get_generation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['prompt'], "test prompt")
        
        # Test clear history
        result = self.backend.clear_history()
        self.assertIn("cleared", result.lower())
        self.assertEqual(len(self.backend.get_generation_history()), 0)
    
    def test_style_presets(self):
        """Test style preset application."""
        original_cfg = self.backend.current_settings['cfg_scale']
        
        result = self.backend.apply_style_preset('photorealistic')
        self.assertIn("Applied photorealistic", result)
        # CFG scale should have changed
        self.assertNotEqual(self.backend.current_settings['cfg_scale'], original_cfg)
        
        # Test invalid preset
        result = self.backend.apply_style_preset('invalid_style')
        self.assertIn("not found", result)
    
    def test_process_image_upload(self):
        """Test image upload processing."""
        # Test with None
        result_img, message = self.backend.process_image_upload(None)
        self.assertIsNone(result_img)
        self.assertIn("No image", message)
        
        # Test with valid image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result_img, message = self.backend.process_image_upload(test_image)
        self.assertIsNotNone(result_img)
        self.assertIn("processed", message.lower())

class TestHandlerFunctions(unittest.TestCase):
    """Test cases for handler functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not BACKEND_AVAILABLE:
            self.skipTest("Backend not available")
    
    def test_handle_generate_click(self):
        """Test generate button handler."""
        result_image, result_info = handle_generate_click(
            prompt="test prompt",
            negative_prompt="",
            steps=20,
            cfg_scale=7.5,
            width=512,
            height=512,
            seed=42,
            sampler="Euler a",
            model="stable-diffusion-v1-5",
            batch_size=1
        )
        
        self.assertIsNotNone(result_image)
        self.assertIsInstance(result_image, np.ndarray)
        self.assertIsInstance(result_info, str)
        self.assertIn("prompt", result_info)
    
    def test_handle_model_change(self):
        """Test model change handler."""
        if backend.available_models:
            model = backend.available_models[0]
            result = handle_model_change(model)
            self.assertIsInstance(result, str)
            self.assertIn("changed to", result)
    
    def test_handle_sampler_change(self):
        """Test sampler change handler."""
        if backend.available_samplers:
            sampler = backend.available_samplers[0]
            result = handle_sampler_change(sampler)
            self.assertIsInstance(result, str)
            self.assertIn("changed to", result)
    
    def test_handle_randomize_seed(self):
        """Test randomize seed handler."""
        new_seed, message = handle_randomize_seed()
        self.assertIsInstance(new_seed, int)
        self.assertIsInstance(message, str)
        self.assertIn("randomized", message.lower())
    
    def test_handle_style_preset_change(self):
        """Test style preset change handler."""
        cfg, steps, sampler, message = handle_style_preset_change('photorealistic')
        self.assertIsInstance(cfg, float)
        self.assertIsInstance(steps, int)
        self.assertIsInstance(sampler, str)
        self.assertIsInstance(message, str)
        self.assertIn("Applied", message)
    
    def test_handle_save_load_settings(self):
        """Test save and load settings handlers."""
        # Test save
        save_result = handle_save_settings()
        self.assertIsInstance(save_result, str)
        self.assertIn("saved", save_result.lower())
        
        # Test load
        load_result = handle_load_settings()
        self.assertIsInstance(load_result, tuple)
        self.assertEqual(len(load_result), 8)  # Should return 8 values
    
    def test_handle_clear_history(self):
        """Test clear history handler."""
        result = handle_clear_history()
        self.assertIsInstance(result, str)
        self.assertIn("cleared", result.lower())
    
    def test_handle_image_upload(self):
        """Test image upload handler."""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result_img, message = handle_image_upload(test_image)
        self.assertIsNotNone(result_img)
        self.assertIsInstance(message, str)
    
    def test_handle_batch_settings_change(self):
        """Test batch settings change handler."""
        result = handle_batch_settings_change(2, 3)
        self.assertIsInstance(result, str)
        self.assertIn("Batch settings", result)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not BACKEND_AVAILABLE:
            self.skipTest("Backend not available")
    
    def test_full_generation_workflow(self):
        """Test a complete generation workflow."""
        # Update settings
        handle_model_change("stable-diffusion-v1-5")
        handle_sampler_change("Euler a")
        
        # Apply style preset
        handle_style_preset_change("photorealistic")
        
        # Generate image
        result_image, result_info = handle_generate_click(
            prompt="A beautiful sunset over mountains",
            negative_prompt="blurry, low quality",
            steps=25,
            cfg_scale=7.0,
            width=512,
            height=512,
            seed=12345,
            sampler="Euler a",
            model="stable-diffusion-v1-5",
            batch_size=1
        )
        
        # Verify results
        self.assertIsNotNone(result_image)
        self.assertIsInstance(result_image, np.ndarray)
        self.assertEqual(result_image.shape, (512, 512, 3))
        self.assertIn("sunset", result_info)
    
    def test_settings_persistence(self):
        """Test settings save and load workflow."""
        # Modify settings
        backend.update_steps(30)
        backend.update_cfg_scale(8.5)
        backend.update_dimensions(768, 768)
        
        # Save settings
        save_result = handle_save_settings()
        self.assertIn("saved", save_result.lower())
        
        # Modify settings again
        backend.update_steps(15)
        backend.update_cfg_scale(5.0)
        
        # Load settings
        load_result = handle_load_settings()
        self.assertIsInstance(load_result, tuple)
        
        # Verify settings were restored
        self.assertEqual(backend.current_settings['steps'], 30)
        self.assertEqual(backend.current_settings['cfg_scale'], 8.5)

def run_tests():
    """Run all tests and return results."""
    if not BACKEND_AVAILABLE:
        print("❌ Backend not available - skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestForgeDreamBackend))
    test_suite.addTest(unittest.makeSuite(TestHandlerFunctions))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)