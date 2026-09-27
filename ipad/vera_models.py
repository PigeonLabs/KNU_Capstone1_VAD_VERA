"""Frozen local VLM and ImageBind adapters (no ground-truth inputs)."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from .vera import build_prompt


class InternVL:
    def __init__(self, root, template, config):
        from transformers import AutoModel, AutoTokenizer
        self.config = config
        self.prompt = build_prompt(template)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True, use_fast=False, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(root), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, local_files_only=True, use_flash_attn=False,
        ).eval().requires_grad_(False).cuda()
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    @torch.inference_mode()
    def predict(self, files):
        images = []
        for file in files:
            with Image.open(file) as im:
                # Upstream dynamic_preprocess(grid_size=1) first performs a PIL resize.
                images.append(self.transform(im.convert('RGB').resize((448, 448))))
        pixels = torch.stack(images).to(device='cuda', dtype=torch.bfloat16)
        return self.model.chat(self.tokenizer, pixels, self.prompt,
                               dict(num_beams=1, max_new_tokens=self.config.max_new_tokens, do_sample=False),
                               num_patches_list=[1]*len(files))


class ImageBind:
    def __init__(self, cache, reference):
        sys.path.insert(0, str((cache/'ImageBind').resolve()))
        # pytorchvideo 0.1.5 imports this old torchvision module name; the function
        # implementation remains in the private replacement module in torchvision.
        import torchvision.transforms._functional_tensor as functional_tensor
        sys.modules.setdefault('torchvision.transforms.functional_tensor', functional_tensor)
        from imagebind.models.imagebind_model import imagebind_huge
        spec = importlib.util.spec_from_file_location('vera_lavad_data', reference/'lavad_data.py')
        self.data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.data)
        self.model = imagebind_huge(pretrained=False)
        state = torch.load(cache/'imagebind_huge.pth', map_location='cpu', weights_only=True, mmap=True)
        self.model.load_state_dict(state)
        self.model.eval().requires_grad_(False).cuda()

    @torch.inference_mode()
    def encode(self, files, segment, samples=10):
        # LAVAD's documented frame-list adapter: uniform endpoint-inclusive sampling,
        # then its original five temporal clips / three spatial crops transformation.
        ids = np.linspace(segment['start'], segment['end']-1,
                          min(samples, segment['end']-segment['start']), dtype=int)
        selected = [str(files[i]) for i in ids]
        pixels = self.data.load_and_transform_video_data([selected], 'cuda')
        feature = self.model({'vision': pixels})['vision'][0].float().cpu().numpy()
        return feature, ids.tolist()
