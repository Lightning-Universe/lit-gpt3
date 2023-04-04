# Lightning GPT3 Component

## Description

This component makes it a breeze to integrate GPT-3 into your Lightning Apps. It wraps the OpenAI API into a  user-friendly class, providing a smooth integration and improved performance.

To use the component you'll need to generate your OpenAI API key. To generate your key, you'll need to sign up for an OpenAI account.

## What is GPT3

GPT-3 ("Generative Pretrained Transformer 3") is a natural language processing (NLP) model developed by OpenAI that can generate human-like text, perform language translation, and answer questions.

## Let's integrate it into an example:

This example showcases how to use the component to enhance the prompts to Stable Diffusion (SD). Using these prompts, Stable Diffusion can generate more detailed and realistic images with the SD algorithm. To run this example save this code as app.py:

```python
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/lit/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'


import lightning as L
import base64, io, os, ldm, pydantic, torch
from lightning_gpt3 import LightningGPT3

# For running on M1/M2
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class PromptEnhancedStableDiffusionServer(L.app.components.PythonServer):
    def __init__(self, cloud_compute, input_type, output_type):
        super().__init__(
            input_type=input_type, output_type=output_type, cloud_compute=cloud_compute
        )
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
        cloud_compute=L.CloudCompute("gpu-fast", disk_size=80),
        input_type=SD_input,
        output_type=SD_output,
    )
)
```

## Installing Lightning

If you don't have Lightning installed yet, install it using the command:

```bash

pip install -U lightning

```

## Run it llocally

Run it locally using

```bash

lightning run app app.py --setup --env OPENAI_API_KEY=<OPENAI_API_KEY>

```

**_NOTE:_**  This process will download the stable diffusion weights from s3, which are 4.27 GB in size. The time it takes will depend on your internet speed

## Run it in the cloud

Run it in the cloud using

```bash

lightning run app app.py --setup --env OPENAI_API_KEY=<OPENAI_API_KEY>  --cloud

```

## Make a request:

To make a request  save this code as client.py:

```python
import base64, io, requests, PIL.Image as Image

if __name__ == "__main__":
    response = requests.post(
        "https://gmohx-01gpksza0c543x0xmzt8wpb55n.litng-ai-03.litng.ai/predict",
        json={"text": "forest-inspired bedroom", "enhacement": True},
    )

    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
    print(response.json()["Enhanced_prompt"])
```

Then, run:

```bash

python client.py

```

## More Examples:

TODO
