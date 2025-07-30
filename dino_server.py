import os
import base64
from io import BytesIO
import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from typing import Tuple
import types
from datetime import datetime

from utils.utils_correspondence import resize

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def preprocess_pil(pil_image: Image.Image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(pil_image).unsqueeze(0)  # [1, 3, H, W]

def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
    def interpolate_pos_encoding(self, tokens: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = tokens.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = tokens.shape[-1]

        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f"Expected {npatch} tokens, got {h0}x{w0} = {h0 * w0}"

        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    return interpolate_pos_encoding

def patch_vit_stride(model: nn.Module, stride: int) -> nn.Module:
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    if stride == patch_size:
        return model

    stride = (stride, stride)
    model.patch_embed.proj.stride = stride
    model.interpolate_pos_encoding = types.MethodType(_fix_pos_enc(patch_size, stride), model)
    return model


set_seed(42)
dino_stride = 16
dino_patch_size = 14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load DINOv2 model and apply stride
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model = patch_vit_stride(dino_model, stride=dino_stride).to(device).eval()

app = Flask(__name__)

@app.route('/get-dino-features', methods=['POST'])
def extract_features():
    #try:
    # Read input JSON and convert to tensor
    data = request.get_json()
    img = Image.open(BytesIO(base64.b64decode(data['image']))).convert("RGB")
    print(type(img))
    img_dino_input = resize(img, target_res=644, resize=True, to_pil=True)
    img_batch = preprocess_pil(img_dino_input)

    # Run model
    with torch.no_grad():
        tokens = dino_model.forward_features(img_batch.cuda())
        features = tokens["x_norm_patchtokens"].permute(2, 0, 1).reshape(1, -1, 40, 40)

    return jsonify({'features': features.detach().cpu().numpy().tolist()})

    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2096)
