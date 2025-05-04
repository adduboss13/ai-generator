import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# मॉडल लोड करें (पहली बार 5GB डाउनलोड होगा)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

def generate_image(prompt):
    # इमेज जनरेट करें
    image = pipe(prompt).images[0]
    return image

# ग्रैडियो इंटरफ़ेस
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="प्रॉम्प्ट डालें (हिंदी/इंग्लिश)"),
    outputs=gr.Image(label="जनरेटेड इमेज"),
    title="AI इमेज जनरेटर (Stable Diffusion)",
    examples=["हिमालय पर बौद्ध मंदिर", "एक रोबोट शेर जंगल में"]
)

interface.launch(server_name="0.0.0.0", server_port=7860)
