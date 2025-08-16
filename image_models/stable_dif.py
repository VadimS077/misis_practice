from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import torch

class StableDiffusionPipelineWrapper:
    def __init__(
        self,
        model_path="runwayml/stable-diffusion-v1-5",
        device='cuda'
    ):
        self.device = device
        
        # Явная загрузка компонентов
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(device)
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None,  # Отключаем фильтр безопасности
        ).to(device)

    @torch.no_grad()
    def __call__(self, prompt,num_inference=30, seed=0, size=512):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            generator=generator,
            width=size,
            height=size,
            num_inference_steps=num_inference
        ).images[0]
        return image