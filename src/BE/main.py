import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display
import random
import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image

#CONSTANTs
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
g_cuda = torch.Generator(device=device) 
seed = random.randint(0, 10000)
g_cuda.manual_seed(seed)
MODEL_PATH = "../assets/model_sd.ckpt"
W=H=256

def load_model_from_config(model_path, device):
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


PIPE = load_model_from_config(MODEL_PATH, device)

def infer(prompt,negative_prompt="", num_samples=1, W=W,H=H):
    guidance_scale = 7.5
    num_inference_steps = 24 
    height = W
    width = H

    with autocast("cuda"), torch.inference_mode():
        images = PIPE(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    return images