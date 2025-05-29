"""
Fixed UI components for Forge Dream extension with proper event handlers.
All interactive components now have backend methods to prevent Gradio errors.
"""

import gradio as gr
from backend import (
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

def create_main_interface():
    """Create the main Forge Dream interface with all components properly connected."""
    
    with gr.Blocks(title="Forge Dream - AI Image Generation", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Forge Dream - Advanced AI Image Generation")
        gr.Markdown("Generate stunning images with state-of-the-art AI models")
        
        with gr.Row():
            # Left column - Input controls
            with gr.Column(scale=1):
                gr.Markdown("## Generation Settings")
                
                # Text inputs
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value=""
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to avoid in the image...",
                    lines=2,
                    value=""
                )
                
                # Model and sampler selection
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=backend.available_models,
                        value=backend.current_settings['model'],
                        interactive=True
                    )
                    
                    sampler_dropdown = gr.Dropdown(
                        label="Sampler",
                        choices=backend.available_samplers,
                        value=backend.current_settings['sampler'],
                        interactive=True
                    )
                
                # Style presets
                style_preset_dropdown = gr.Dropdown(
                    label="Style Preset",
                    choices=["photorealistic", "artistic", "anime", "fantasy"],
                    value=None,
                    interactive=True
                )
                
                # Generation parameters
                with gr.Row():
                    steps_slider = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=150,
                        value=backend.current_settings['steps'],
                        step=1,
                        interactive=True
                    )
                    
                    cfg_scale_slider = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=30.0,
                        value=backend.current_settings['cfg_scale'],
                        step=0.1,
                        interactive=True
                    )
                
                # Image dimensions
                with gr.Row():
                    width_slider = gr.Slider(
                        label="Width",
                        minimum=64,
                        maximum=2048,
                        value=backend.current_settings['width'],
                        step=64,
                        interactive=True
                    )
                    
                    height_slider = gr.Slider(
                        label="Height",
                        minimum=64,
                        maximum=2048,
                        value=backend.current_settings['height'],
                        step=64,
                        interactive=True
                    )
                
                # Seed controls
                with gr.Row():
                    seed_number = gr.Number(
                        label="Seed",
                        value=backend.current_settings['seed'],
                        precision=0,
                        interactive=True
                    )
                    
                    randomize_seed_btn = gr.Button(
                        "🎲 Randomize",
                        size="sm",
                        variant="secondary"
                    )
                
                # Batch settings
                with gr.Row():
                    batch_size_slider = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=8,
                        value=backend.current_settings['batch_size'],
                        step=1,
                        interactive=True
                    )
                    
                    batch_count_slider = gr.Slider(
                        label="Batch Count",
                        minimum=1,
                        maximum=10,
                        value=backend.current_settings['batch_count'],
                        step=1,
                        interactive=True
                    )
                
                # Main generate button
                generate_btn = gr.Button(
                    "🚀 Generate Image",
                    variant="primary",
                    size="lg"
                )
                
                # Settings management
                gr.Markdown("## Settings Management")
                with gr.Row():
                    save_settings_btn = gr.Button("💾 Save Settings", size="sm")
                    load_settings_btn = gr.Button("📁 Load Settings", size="sm")
                    clear_history_btn = gr.Button("🗑️ Clear History", size="sm")
            
            # Right column - Output and results
            with gr.Column(scale=1):
                gr.Markdown("## Generated Image")
                
                # Output image
                output_image = gr.Image(
                    label="Generated Image",
                    type="numpy",
                    interactive=False
                )
                
                # Generation info
                generation_info = gr.Textbox(
                    label="Generation Info",
                    lines=10,
                    interactive=False,
                    show_copy_button=True
                )
                
                # Status messages
                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_copy_button=False
                )
                
                # Image upload for img2img (future feature)
                gr.Markdown("## Image Upload (img2img)")
                uploaded_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    interactive=True
                )
                
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False
                )
        
        # Event handlers - Connect all components to backend functions
        
        # Generate button click
        generate_btn.click(
            fn=handle_generate_click,
            inputs=[
                prompt_input,
                negative_prompt_input,
                steps_slider,
                cfg_scale_slider,
                width_slider,
                height_slider,
                seed_number,
                sampler_dropdown,
                model_dropdown,
                batch_size_slider
            ],
            outputs=[output_image, generation_info]
        )
        
        # Model dropdown change
        model_dropdown.change(
            fn=handle_model_change,
            inputs=[model_dropdown],
            outputs=[status_message]
        )
        
        # Sampler dropdown change
        sampler_dropdown.change(
            fn=handle_sampler_change,
            inputs=[sampler_dropdown],
            outputs=[status_message]
        )
        
        # Randomize seed button
        randomize_seed_btn.click(
            fn=handle_randomize_seed,
            inputs=[],
            outputs=[seed_number, status_message]
        )
        
        # Style preset change
        style_preset_dropdown.change(
            fn=handle_style_preset_change,
            inputs=[style_preset_dropdown],
            outputs=[cfg_scale_slider, steps_slider, sampler_dropdown, status_message]
        )
        
        # Settings management buttons
        save_settings_btn.click(
            fn=handle_save_settings,
            inputs=[],
            outputs=[status_message]
        )
        
        load_settings_btn.click(
            fn=handle_load_settings,
            inputs=[],
            outputs=[
                steps_slider,
                cfg_scale_slider,
                width_slider,
                height_slider,
                seed_number,
                sampler_dropdown,
                model_dropdown,
                status_message
            ]
        )
        
        clear_history_btn.click(
            fn=handle_clear_history,
            inputs=[],
            outputs=[status_message]
        )
        
        # Image upload handler
        uploaded_image.change(
            fn=handle_image_upload,
            inputs=[uploaded_image],
            outputs=[uploaded_image, upload_status]
        )
        
        # Batch settings change
        batch_size_slider.change(
            fn=handle_batch_settings_change,
            inputs=[batch_size_slider, batch_count_slider],
            outputs=[status_message]
        )
        
        batch_count_slider.change(
            fn=handle_batch_settings_change,
            inputs=[batch_size_slider, batch_count_slider],
            outputs=[status_message]
        )
    
    return interface

def create_advanced_interface():
    """Create an advanced interface with additional features."""
    
    with gr.Blocks(title="Forge Dream Advanced", theme=gr.themes.Glass()) as advanced_interface:
        gr.Markdown("# 🔬 Forge Dream - Advanced Controls")
        
        with gr.Tabs():
            # Basic generation tab
            with gr.TabItem("Basic Generation"):
                basic_interface = create_main_interface()
            
            # Advanced controls tab
            with gr.TabItem("Advanced Controls"):
                gr.Markdown("## Advanced Generation Parameters")
                
                with gr.Row():
                    with gr.Column():
                        # Advanced sampling parameters
                        eta_slider = gr.Slider(
                            label="Eta (DDIM)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.01
                        )
                        
                        ddim_steps = gr.Slider(
                            label="DDIM Steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1
                        )
                        
                        # Conditioning parameters
                        conditioning_scale = gr.Slider(
                            label="Conditioning Scale",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )
                    
                    with gr.Column():
                        # Output controls
                        output_format = gr.Dropdown(
                            label="Output Format",
                            choices=["PNG", "JPEG", "WEBP"],
                            value="PNG"
                        )
                        
                        quality_slider = gr.Slider(
                            label="Quality (for JPEG)",
                            minimum=1,
                            maximum=100,
                            value=95,
                            step=1
                        )
                        
                        # Metadata options
                        include_metadata = gr.Checkbox(
                            label="Include Generation Metadata",
                            value=True
                        )
            
            # History tab
            with gr.TabItem("Generation History"):
                gr.Markdown("## Generation History")
                
                history_gallery = gr.Gallery(
                    label="Previous Generations",
                    show_label=True,
                    elem_id="history_gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
                
                refresh_history_btn = gr.Button("🔄 Refresh History")
                export_history_btn = gr.Button("📤 Export History")
                
                # Connect history refresh
                refresh_history_btn.click(
                    fn=lambda: backend.get_generation_history(),
                    inputs=[],
                    outputs=[history_gallery]
                )
    
    return advanced_interface

# Create the interfaces
main_interface = create_main_interface()
advanced_interface = create_advanced_interface()

# Export for use in main module
__all__ = ['main_interface', 'advanced_interface', 'create_main_interface', 'create_advanced_interface']