# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml


import lightning as L
import torch, os, io, base64
from lightning.app.components import Image, serve, Text
from ldm.lightning import LightningStableDiffusion, PromptDataset
from lightning_gpt3 import LightningGPT3


# For running on M1/M2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class StableDiffusionServer(serve.PythonServer):
    def __init__(self, cloud_compute, input_type=Text, output_type=Image):
        super().__init__(input_type=input_type, output_type=output_type, cloud_compute=cloud_compute)
        self._model = None
        self._gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))

    def setup(self):
        os.system(
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        )

        self._trainer = L.Trainer(
            accelerator="auto", devices=1, precision=16 if torch.cuda.is_available() else 32, enable_progress_bar=False
        )

        self._model = LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=512,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request: Text):
        prompt = "Describe a " + request.text + " picture"
        enhanced_prompt = self._gpt3.generate(prompt=prompt, max_tokens=40)
        image = self._trainer.predict(self._model, torch.utils.data.DataLoader(PromptDataset([enhanced_prompt])))[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


app = L.LightningApp(StableDiffusionServer(cloud_compute=L.CloudCompute("gpu-fast", disk_size=80)))
