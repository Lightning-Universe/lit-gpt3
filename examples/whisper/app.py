# !pip install setuptools-rust
# !pip install git+https://github.com/openai/whisper.git
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'
# !sudo apt update && sudo apt install ffmpeg


import lightning as L
import torch, os, io, base64, whisper
from lightning.app.components import Image, serve
from pydantic import BaseModel

from lightning_gpt3 import LightningGPT3


class Text(BaseModel):
    text: str


class StableDiffusionServer(serve.PythonServer):
    def __init__(
        self,
        input_type=Text,
        output_type=Text,
    ):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            cloud_compute=L.CloudCompute("gpu-fast"),
        )

        self._model = None
        self._gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))

    def setup(self):
        self._model = whisper.load_model("base")

    def predict(self, request):
        file = open("sample" + ".mp3", "wb")
        file.write(base64.b64decode(request.text))
        file.close()

        result = self._model.transcribe("sample.mp3")

        answer = self._gpt3.generate(prompt=result["text"], max_tokens=100)

        return {"text": str(answer)}


app = L.LightningApp(StableDiffusionServer())
