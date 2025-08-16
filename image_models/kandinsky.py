from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers import Kandinsky3Pipeline
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import T5Tokenizer
import torch

class KandinskyPipeline:
    def __init__(
        self,
        prior_path="kandinsky-community/kandinsky-2-2-prior",
        decoder_path="kandinsky-community/kandinsky-2-2-decoder",
        device="cuda"
    ):
        self.device = device

     
        self.prior = KandinskyV22PriorPipeline.from_pretrained(
            prior_path,
            torch_dtype=torch.float16
        ).to(device)


        self.decoder = KandinskyV22Pipeline.from_pretrained(
            decoder_path,
            torch_dtype=torch.float16
        ).to(device)

        self.prior.set_progress_bar_config(disable=True)
        self.decoder.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, prompt, num_inference=35, seed=0, size=768):
        generator = torch.Generator(device=self.device).manual_seed(seed)


        prior_output = self.prior(prompt, generator=generator)
        image_embeds = prior_output.image_embeds
        negative_embeds = prior_output.negative_image_embeds


        images = self.decoder(
            image_embeds=image_embeds,
            negative_image_embeds=negative_embeds,
            height=size,
            width=size,
            num_inference_steps=num_inference,
            generator=generator
        ).images

        return images[0]


class Kandinsky3:
    def __init__(
        self,
        model_path="kandinsky-community/kandinsky-3",
        device="cuda"
    ):
        self.device = device

    
        self.pipe = Kandinsky3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, prompt, seed=42, num_inference=25, size=1024):
        generator = torch.Generator(self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            height=size,
            width=size,
            num_inference_steps=num_inference,
            generator=generator
        ).images[0]

        return image
