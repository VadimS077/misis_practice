from PIL import Image

from image_models.stable_dif_xl import SDXLPipeline
from image_models.kandinsky import KandinskyPipeline, Kandinsky3
import torch
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

model_path = 'tencent/Hunyuan3D-2'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)

image_generator = Kandinsky3()
prompt="realistic,for 3d render, a single bottle of Fanta, white background, no other objects "
image = image_generator(prompt,size=1024,num_inference=50)
image.save("fanta.png")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)
else:
    rgba_image = image.convert("RGBA")

mesh = pipeline_shapegen(image=image,
                         num_inference_steps=50,
    octree_resolution=150,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh')[0]
mesh = pipeline_texgen(mesh, image=image)
mesh.export('fanta.glb')