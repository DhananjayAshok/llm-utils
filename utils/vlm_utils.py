import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig
from utils import log_error
import os



def get_single_vlm_message_list(text: str, images: list[Image.Image]) -> dict:
    content = [{"type": "text", "text": text}]
    for img in images:
        if not os.path.exists(img):
            log_error(f"Image path {img} does not exist.")
        content.append({"type": "image", "image": img})
    return [{"role": "user", "content": content}]


def get_vlm_message_list(df_or_ds, text_column="input", image_column="image"):
    messages = []
    for item in range(len(df_or_ds)):
        text = df_or_ds[text_column][item]
        images = df_or_ds[image_column][item]
        messages.append(get_single_vlm_message_list(text, images))
    return messages