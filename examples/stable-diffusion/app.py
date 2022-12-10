# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml


import lightning as L
import torch, os, io, base64, pydantic
from lightning.app.components import Image, serve
from ldm.lightning import LightningStableDiffusion, PromptDataset
from lightning_gpt3 import LightningGPT3


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


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
        os.system(
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        )
        os.system("echo *.ckpt > .lightningignore ")

        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
            inference_mode=torch.cuda.is_available(),
        )

        self._model = LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=768,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request: Text):
        print(self._trainer.strategy.root_device.type)
        prompt = "Describe a " + request.text + " picture"
        enhanced_prompt = self._gpt3.generate(prompt=prompt, max_tokens=40)[2::]
        image = self._trainer.predict(self._model, torch.utils.data.DataLoader(PromptDataset([enhanced_prompt])),)[
            0
        ][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


app = L.LightningApp(StableDiffusionServer())
