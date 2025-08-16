
### Install Requirements

Please install Pytorch via the [official](https://pytorch.org/) site. Then install the other requirements via

```bash
pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

For testing add some prompts on config.json and then launch generator.py

```bash
python generator.py --config config.json --output_dir outputs/ --model sdxl
```
