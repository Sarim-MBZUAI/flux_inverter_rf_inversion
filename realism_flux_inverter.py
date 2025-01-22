import argparse
import torch
from pipeline_flux_rf_inversion import FluxRFInversionPipeline
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import os
from pathlib import Path


def clean_filename_to_prompt(filename):
    # Remove extension and convert to prompt
    prompt = os.path.splitext(filename)[0]
    # Replace underscores with spaces
    prompt = prompt.replace('_', ' ')
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Run Flux RF Inversion Pipeline")
    parser.add_argument("--model", type=str, default="/ephemeral/shashmi/FLUX.1-dev", help="Model name or path")
    parser.add_argument("--lora_path", type=str,default="/shared/shashmi/flux-RealismLora/lora.safetensors", help="Path to the realism LoRA weights")
    parser.add_argument("--input_base_dir", type=str, required=True, help="Base directory containing category folders")
    parser.add_argument("--output_base_dir", type=str, default='invflux', help="Base directory for output images")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.95, help="Strength parameter")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter")
    parser.add_argument("--eta", type=float, default=0.9, help="Eta parameter")
    parser.add_argument("--start_timestep", type=int, default=0, help="Start timestep")
    parser.add_argument("--stop_timestep", type=int, default=6, help="Stop timestep")
    parser.add_argument("--use_img2img", action="store_true", help="Use FluxImg2ImgPipeline instead of RF Inversion")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per input image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0', 'cuda:1')")

    args = parser.parse_args()

    # Create base output directory if it doesn't exist
    os.makedirs(args.output_base_dir, exist_ok=True)

    # Check CUDA availability and set device
    if not torch.cuda.is_available() and args.device.startswith("cuda:0"):
        print("CUDA is not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device
    
    # Initialize pipeline
    if args.use_img2img:
        pipe = FluxImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    else:
        pipe = FluxRFInversionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    
    pipe = pipe.to(device)

    # Load the LoRA weights using PEFT
    try:
        # Load PEFT configuration and weights
        # pipe.text_encoder = PeftModel.from_pretrained(
        #     pipe.text_encoder,
        #     args.lora_path,
        #     adapter_name="realism_lora"
        # )
        pipe.load_lora_weights(args.lora_path, weight_name="lora.safetensors")
        print(f"Successfully loaded LoRA weights from {args.lora_path}")

    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        return

    # Supported image extensions
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    # Get all category directories
    category_dirs = [d for d in os.listdir(args.input_base_dir) 
                    if os.path.isdir(os.path.join(args.input_base_dir, d))]

    for category in category_dirs:
        input_category_dir = os.path.join(args.input_base_dir, category)
        output_category_dir = os.path.join(args.output_base_dir, f"inv_{category}")
        
        # Create output directory for this category
        os.makedirs(output_category_dir, exist_ok=True)
        
        print(f"\nProcessing category: {category}")
        
        # Process each image in the category directory
        for filename in os.listdir(input_category_dir):
            if Path(filename).suffix.lower() in supported_extensions:
                try:
                    input_path = os.path.join(input_category_dir, filename)
                    
                    # Generate prompt from filename
                    prompt = clean_filename_to_prompt(filename)
                    
                    # Load image
                    init_image = Image.open(input_path)
                    
                    # Store original size
                    original_size = init_image.size
                    
                    # Convert RGBA to RGB if needed
                    if init_image.mode == 'RGBA':
                        background = Image.new('RGB', init_image.size, (255, 255, 255))
                        background.paste(init_image, mask=init_image.split()[3])
                        init_image = background
                    elif init_image.mode != 'RGB':
                        init_image = init_image.convert('RGB')
                    
                    # Resize for processing
                    init_image_resized = init_image.resize((1024, 1024))
                    
                    # Use the same filename for output
                    output_base = os.path.join(output_category_dir, filename)
                    
                    print(f"Processing {category}/{filename}")
                    print(f"Using prompt: {prompt}")
                    print(f"Original dimensions: {original_size}")

                    for i in range(args.num_images):
                        # Set up generator for reproducibility
                        generator = torch.Generator(device=device).manual_seed(i)

                        kwargs = {
                            "gamma": args.gamma,
                            "eta": args.eta,
                            "start_timestep": args.start_timestep,
                            "stop_timestep": args.stop_timestep
                        } if not args.use_img2img else dict({})

                        # Generate image
                        generated_image = pipe(
                            prompt=prompt,
                            prompt_2=prompt,  # Using same prompt for prompt_2
                            image=init_image_resized,
                            num_inference_steps=args.num_inference_steps,
                            strength=args.strength,
                            guidance_scale=args.guidance_scale,
                            generator=generator,
                            **kwargs,
                        ).images[0]

                        # Resize back to original dimensions
                        generated_image = generated_image.resize(original_size, Image.Resampling.LANCZOS)

                        # Save with original filename (adding index if multiple images per input)
                        if args.num_images > 1:
                            output_path = f"{os.path.splitext(output_base)[0]}_{i}{os.path.splitext(output_base)[1]}"
                        else:
                            output_path = output_base
                            
                        generated_image.save(output_path)
                        print(f"Saved output image as {output_path} with original dimensions {original_size}")

                        # Clear CUDA cache after each generation
                        if device.startswith("cuda"):
                            torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing {category}/{filename}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()