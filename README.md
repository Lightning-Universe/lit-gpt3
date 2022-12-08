# Lightning GPT3 Component 

## Description

This component makes it a breeze to integrate GPT-3 into your Lightning Apps. It wraps the OpenAI API into a  user-friendly class, providing a smooth integration and improved performance.

To use the component you'll need to generate your OpenAI API key. To generate your key, you'll need to sign up for an OpenAI account.

## What is GPT3
   
    GPT-3 ("Generative Pretrained Transformer 3") is a natural language processing (NLP) model developed by OpenAI that can generate human-like text, perform language translation, and answer questions.
  
## Let's integrate it into an example:

This example showcases how to use the component to enhance the prompts in the Stable Diffusion (SD). Using these prompts, Stable Diffusion can generate more detailed and realistic images with the SD algorithm. To run this example save this code as app.py:


``` python 
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
    def __init__(self, input_type=Text,output_type=Image):
        super().__init__(
            input_type=input_type,
            output_type=output_type,
            cloud_compute=L.CloudCompute("gpu-fast", shm_size=512))
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
    print
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
```


Then, run:
```  bash

python client.py

```

## More Examples:
TODO

