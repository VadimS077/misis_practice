from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch

class SDXLPipeline:
    def __init__(
        self,
        model_path="stabilityai/stable-diffusion-xl-base-1.0",
        device='cuda'
    ):
        self.device = device
        
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16
        ).to(device)
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, prompt, num_inference=35, seed=0, size=1024):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            generator=generator,
            width=size,
            height=size,
            num_inference_steps=num_inference
        ).images[0]