import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig
from utils import log_error

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def infer_vlm_kind(model_name=None, config=None):
    """
    Infer the kind of VLM based on the model name.
    """
    if config is None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    architecture = config.architectures[0]
    if architecture in ["LlavaNextForConditionalGeneration"]:
        return "llava-next"
    elif architecture in ["Qwen2_5_VLForConditionalGeneration"]:
        return "qwen2.5"
    elif architecture in ["InternVLChatModel"]:
        return "internvl"
    elif architecture in ["Ovis2_5"]:
        return "ovis"   
    elif architecture in ["Qwen3VLForConditionalGeneration"]:
        return "qwen3"
    else:
        log_error(f"Unrecognized Model Kind: {model_name} with architecture {architecture}")


def get_vlm_text(vlm_kind, input_texts):
    if vlm_kind in ["llava-next"]:
        for i, input_text in enumerate(input_texts):
            input_texts[i] = get_single_vlm_text("llava-next", input_text)
        if not isinstance(input_texts, list):
            input_texts = input_texts.tolist()
        return input_texts
    elif vlm_kind in ["qwen2.5", "qwen3"]:
        for i, input_text in enumerate(input_texts):
            input_texts[i] = get_single_vlm_text(vlm_kind, input_text)
        if not isinstance(input_texts, list):
            input_texts = input_texts.tolist()
        return input_texts
    elif vlm_kind in ["internvl"]:
        raise NotImplementedError("This is not implemented yet, need to figure out how to handle internvl inputs")

def get_single_vlm_text(vlm_kind, input_text):
    if vlm_kind in ["llava-next"]:
        if "<image>" in input_text:
            input_text = input_text.replace("<image>", "")
        input_text = "[INST] <image>\n" + input_text + "[/INST]"
        return input_text
    elif vlm_kind in ["qwen2.5", "qwen3"]:
        input_text = "<|im_start|>user\n<vision_start|><|image_pad|><|vision_end|>\n" + input_text + "\n<|im_end|><|im_start|>assistant\n"
        return input_text
    elif vlm_kind in ["internvl"]:
        raise NotImplementedError("This is not implemented yet, need to figure out how to handle internvl inputs")
    else:
        log_error(f"Unrecognized VLM Kind: {vlm_kind}")


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_intern_vl_pixels(pil_image, input_size=448, max_num=12):
    image = pil_image
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

