# !pip install 'git+https://github.com/Lightning-AI/lightning-diffusion-component.git'
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'

import lightning as L
import torch, os, io, base64, diffusers, pydantic
from lightning.app.components import Image, serve
from lightning_diffusion import download_from_lightning_cloud
from lightning_gpt3 import LightningGPT3


class Text(pydantic.BaseModel):
    text: str


class StableDiffusionServer(serve.PythonServer):
    def __init__(self, input_type=Text, output_type=Image):
        super().__init__(
            input_type=input_type, output_type=output_type, cloud_compute=L.CloudCompute("gpu-fast", shm_size=512)
        )
        self._model = None
        self._gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))

    def setup(self):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self._model = diffusers.StableDiffusionPipeline.from_pretrained("model").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def predict(self, request: Text):
        prompt = "Describe a " + request.text + " picture"
        enhanced_prompt = self._gpt3.generate(prompt=prompt, max_tokens=40)[2::]
        image = self._model(enhanced_prompt)[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


app = L.LightningApp(StableDiffusionServer())

app = L.LightningApp(StableDiffusionServer())
