import requests
import io
from PIL import Image
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

API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def generate_image_caption(image_path):
    # # Diffusion pipeline
    device = torch.device("cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    #     "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
    # )
    # sd_pipe = sd_pipe.to(device)
    #
    # pipeline = DiffusionPipeline.from_pretrained(
    #     "lambdalabs/sd-image-variations-diffusers"
    # )
    #
    # # Image transformations
    # img_transforms = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (224, 224),
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #             antialias=False,
    #         ),
    #         transforms.Normalize(
    #             [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    #         ),
    #     ]
    # )
    #
    # # Image-to-image
    # with Image.open(image_path) as img:
    #     img_tensor = img_transforms(img).to(device).unsqueeze(0)
    #     out = sd_pipe(img_tensor, guidance_scale=3)
    #     out["images"][0].save("img1.jpg")

    # Blip image captioning
    raw_image = Image.open(image_path).convert("RGB")

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)

    # Conditional image captioning

    AI_Intervention = "High"
    Design = "Modern"
    Daytime = "Morning"
    Lighting_Style = 'colored'
    color_theme = "any"
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Stable diffusion pipeline
    prompt = (
        f"Give me a realistic and complete image without changing the angle of {caption} "
        f"which room type: , AI Intervention: {AI_Intervention}, and Lighting style is: {Lighting_Style}"
        f"and Design style: {Design} and Daytime is : {Daytime} and colortheme is : {color_theme}"
    )

    api_response = query({"inputs": prompt})
    image = Image.open(io.BytesIO(api_response))
    image.save("result3.jpg")


generate_image_caption("C:\Master\exterior.jpg")



