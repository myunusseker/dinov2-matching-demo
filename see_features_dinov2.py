import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['INFERENCE'] = '1'
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import requests
from pixel_matcher import PixelMatcher

def resize(img, target_res=224, resize=True, to_pil=True, edge=False, background=0):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        canvas += background
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
            canvas += background
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


def pil_to_base64_str(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def get_dino_features(img=None):
    response = requests.post(
        "http://localhost:2096/get-dino-features",
        json={
            'image': pil_to_base64_str(img)
        }
    )
    features_dino = torch.tensor(response.json()['features']).cuda()
    norms_dino = torch.linalg.norm(features_dino, dim=1, keepdim=True)
    features_dino = features_dino / (norms_dino + 1e-8)
    return features_dino


img_size = 640
img1_path = './flower1.png' # path to the source image
img1 = resize(Image.open(img1_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True, background=255)

img2_path = './flower2.png' # path to the source image
img2 = resize(Image.open(img2_path).convert('RGB'), target_res=img_size, resize=True, to_pil=True, background=255)

feat1_dino = get_dino_features(img=img1)
feat2_dino = get_dino_features(img=img2)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

demo = PixelMatcher([img1, img2], torch.cat([feat1_dino, feat2_dino], dim=0), img_size)
demo._click_and_match_one(fig_size=5)

