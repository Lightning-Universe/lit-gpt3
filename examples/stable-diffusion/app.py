# !pip install 'git+https://github.com/Lightning-AI/lightning-diffusion-component.git'
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'
import base64
import io
import os

import lightning as L
import torch
from diffusers import StableDiffusionPipeline
from lightning.app.components import Image
from lightning.app.components.serve import PythonServer
from lightning_diffusion import download_from_lightning_cloud
from pydantic import BaseModel

from lightning_gpt3 import LightningGPT3


class Text(BaseModel):
    text: str


class StableDiffusionServer(PythonServer):
    def __init__(
        self,
        input_type=Text,
        output_type=Image,
        enhance_prompt_fn=None,
    ):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            cloud_compute=L.CloudCompute("gpu-fast", shm_size=512),
        )

        self._model = None
        self.enhance_prompt_fn = enhance_prompt_fn

    def setup(self):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self._model = StableDiffusionPipeline.from_pretrained("model").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def predict(self, request: Text):
        prompt = "Describe a " + request.text + " picture"
        enhanced_prompt = self.enhance_prompt_fn(prompt)

        # FIXME: Debugging
        print("Original prompt:", request.text)
        print("Enhanced prompt:", enhanced_prompt)

        image = self._model(prompt)[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{img_str}"}


class Root(L.LightningFlow):
    def __init__(self):
        super().__init__()

        self.gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))
        self.stable_diffusion = StableDiffusionServer(enhance_prompt_fn=self.gpt3.generate)

    def configure_layout(self):
        return {"name": "endpoint", "content": self.stable_diffusion.url}

    def run(self):
        self.stable_diffusion.run()


app = L.LightningApp(Root())
