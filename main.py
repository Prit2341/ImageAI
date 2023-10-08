import os
import requests
from PIL import Image
import torch
from torchvision import transforms
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImageVariationPipeline,
)

def generate_image_caption(image_path):
    # Diffusion pipeline
    device = torch.device("cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
    )
    sd_pipe = sd_pipe.to(device)

    pipeline = DiffusionPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers"
    )

    # Image transformations
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            ),
        ]
    )

    # Image-to-image
    with Image.open(image_path) as img:
        img_tensor = img_transforms(img).to(device).unsqueeze(0)
        out = sd_pipe(img_tensor, guidance_scale=3)
        out["images"][0].save("img1.jpg")

    # Blip image captioning
    raw_image = Image.open(image_path).convert("RGB")
    
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    
    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Stable diffusion pipeline
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    pipe = pipe.to(device)

    Room = "Living Room"
    AI_Intervention = "High"
    Mode = "Redesign"
    Design = "Modern"
    prompt = (
        f"Give me a realistic and complete image of {caption} "
        f"which room type: {Room}, AI Intervention: {AI_Intervention}, "
        f"Mode: {Mode} and Design style: {Design}"
    )
    image = pipe(prompt).images[0]
    image.save("result3.jpg")



generate_image_caption("C:\Master\First.jpg")


