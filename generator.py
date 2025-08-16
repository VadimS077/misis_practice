import argparse
import json
import os
import random
from PIL import Image
import torch

from image_models.kandinsky import Kandinsky3
from image_models.stable_dif_xl import SDXLPipeline
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D meshes from multiple prompts using image models and Hunyuan3D.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON file with prompts and settings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--model", type=str, default="sdxl", choices=["kandinsky", "sdxl"],
                        help="Image generation model to use: 'kandinsky' or 'sdxl'")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dir(args.output_dir)


    octree_resolution = config.get("octree_resolution", 150)
    picture_size = config.get("picture_size", 1024)
    num_inference = config.get("num_inference", 50)
    prompts = config.get("prompts", [])

 
    if args.model == "kandinsky":
        print("Using Kandinsky 3 model for image generation.")
        image_generator = Kandinsky3()
    elif args.model == "sdxl":
        print("Using Stable Diffusion XL model for image generation.")
        image_generator = SDXLPipeline()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Other models
    rembg = BackgroundRemover()
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

    for entry in prompts:
        tmp = entry["text"]
        prompt = f"realistic,for 3d render,{tmp}, white background, no other objects"
        name = entry["output_basename"]

        seed = random.randint(1, 10**6)
        print(f"\nPrompt: {prompt}")
        print(f"Generating with seed: {seed}")

        
        image = image_generator(prompt, size=picture_size, num_inference=num_inference, seed=seed)

        image_path = os.path.join(args.output_dir, f"{name}.png")
        image.save(image_path)
        print(f"Saved image to {image_path}")

   
        if image.mode == "RGB":
            image = rembg(image)
        else:
            image = image.convert("RGBA")

  
        print(f"Generating mesh for {name}...")
        mesh = pipeline_shapegen(
            image=image,
            num_inference_steps=50,
            octree_resolution=octree_resolution,
            num_chunks=20000,
            generator=torch.manual_seed(seed),
            output_type="trimesh"
        )[0]


        mesh = pipeline_texgen(mesh, image=image)
        mesh_path = os.path.join(args.output_dir, f"{name}.glb")
        mesh.export(mesh_path)
        print(f"Saved mesh to {mesh_path}")

    print("\nAll prompts processed.")


if __name__ == "__main__":
    #usage python generator.py --config config.json --output_dir outputs/ --model sdxl

    main()
