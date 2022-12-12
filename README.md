# Lightning GPT3 Component 

## Description

This component makes it a breeze to integrate GPT-3 into your Lightning Apps. It wraps the OpenAI API into a  user-friendly class, providing a smooth integration and improved performance.

To use the component you'll need to generate your OpenAI API key. To generate your key, you'll need to sign up for an OpenAI account.

## What is GPT3
GPT-3 ("Generative Pretrained Transformer 3") is a natural language processing (NLP) model developed by OpenAI that can generate human-like text, perform language translation, and answer questions.
    
  
## Let's integrate it into an example:

This example showcases how to use the component to enhance the prompts in the Stable Diffusion (SD). Using these prompts, Stable Diffusion can generate more detailed and realistic images with the SD algorithm. To run this example save this code as app.py:


``` python 
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git@readme'
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml


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
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        )

        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
            inference_mode=False,
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
        enhanced_prompt = self._gpt3.generate(prompt=prompt, max_tokens=40)[2::]
        with torch.no_grad():
            image = self._trainer.predict(self._model, torch.utils.data.DataLoader(PromptDataset([enhanced_prompt])))[
                0
            ][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


app = L.LightningApp(StableDiffusionServer())

```


## Installing Lightning
If you don't have Lightning installed yet, install it using the command:

``` bash

pip install -U lightning

```

## Run it llocally 

Run it locally using
```  bash

lightning run app app.py --setup --env OPENAI_API_KEY=<OPENAI_API_KEY>  

```


## Run it in the cloud

Run it in the cloud using
```  bash

lightning run app app.py --setup --env OPENAI_API_KEY=<OPENAI_API_KEY>  --cloud 

```


## Make a request:
To make a request  save this code as client.py:

``` python 
import base64, io, requests, PIL.Image as Image

if __name__ == "__main__":
    response = requests.post(
        "YOUR_PREDICT_URL",
        json={"text": "YOUR_PROMPT"},
    )
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
```


Then, run:
```  bash

python client.py

```

## More Examples:
TODO

