# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'


import base64
import io
import os

import ldm
import lightning as L
import pydantic
import torch

from lightning_gpt3 import LightningGPT3

# For running on M1/M2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class PromptEnhancedStableDiffusionServer(L.app.components.PythonServer):
    def __init__(self, cloud_compute, input_type, output_type):
        super().__init__(input_type=input_type, output_type=output_type, cloud_compute=cloud_compute)
        self._model = None
        self._gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))

    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        os.system(cmd)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=device,
            steps=40,
            use_deepspeed=False,
        )

    def predict(self, request):
        if request.enhacement:
            prompt = "Describe a " + request.text + " picture"
            prompt = self._gpt3.generate(prompt=prompt, max_tokens=75)
        else:
            prompt = request.text
        image = self._model.predict_step(prompts=[prompt], batch_idx=0)[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}", "Enhanced_prompt": prompt}


class SD_output(pydantic.BaseModel):
    image: str
    Enhanced_prompt: str


class SD_input(pydantic.BaseModel):
    text: str
    enhacement: bool


app = L.LightningApp(
    PromptEnhancedStableDiffusionServer(
        cloud_compute=L.CloudCompute("gpu-fast", disk_size=80), input_type=SD_input, output_type=SD_output
    )
)
