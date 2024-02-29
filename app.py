import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import spaces
import os
from PIL import Image

SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", "0") == "1"

# Constants
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
checkpoints = {
    "1-Step" : ["sdxl_lightning_1step_unet_x0.safetensors", 1],
    "2-Step" : ["sdxl_lightning_2step_unet.safetensors", 2],
    "4-Step" : ["sdxl_lightning_4step_unet.safetensors", 4],
    "8-Step" : ["sdxl_lightning_8step_unet.safetensors", 8],
}


# Ensure model and scheduler are initialized in GPU-enabled function
if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")

if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to("cuda")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda")
        has_nsfw_concepts = safety_checker(
            images=[images],
            clip_input=safety_checker_input.pixel_values.to("cuda")
        )

        return images, has_nsfw_concepts

# Function 
@spaces.GPU(enable_queue=True)
def generate_image(prompt, ckpt):

    checkpoint = checkpoints[ckpt][0]
    num_inference_steps = checkpoints[ckpt][1] 

    if num_inference_steps==1:
        # Ensure sampler uses "trailing" timesteps and "sample" prediction type for 1-step inference.
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample")
    else:
        # Ensure sampler uses "trailing" timesteps.
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
    pipe.unet.load_state_dict(load_file(hf_hub_download(repo, checkpoint), device="cuda"))
    results = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=0)

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(results.images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512))
        return images[0]
    return results.images[0]



# Gradio Interface
description = """
This demo utilizes the SDXL-Lightning model by ByteDance, which is a lightning-fast text-to-image generative model capable of producing high-quality images in 4 steps.
As a community effort, this demo was put together by AngryPenguin. Link to model: https://huggingface.co/ByteDance/SDXL-Lightning
"""

with gr.Blocks(css="style.css") as demo:
    gr.HTML("<h1><center>Text-to-Image with SDXL-Lightning ⚡</center></h1>")
    gr.Markdown(description)
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(label='Enter you image prompt:', scale=8)
            ckpt = gr.Dropdown(label='Select inference steps',choices=['1-Step', '2-Step', '4-Step', '8-Step'], value='4-Step', interactive=True)
            submit = gr.Button(scale=1, variant='primary')
    img = gr.Image(label='SDXL-Lightning Generated Image')

    prompt.submit(fn=generate_image,
                 inputs=[prompt, ckpt],
                 outputs=img,
                 )
    submit.click(fn=generate_image,
                 inputs=[prompt, ckpt],
                 outputs=img,
                 )
    
demo.queue().launch()