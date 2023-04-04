# !pip install nicegui
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !pip install 'git+https://github.com/Lightning-AI/lightning-diffusion-component.git'
# !pip install 'git+https://github.com/Lightning-AI/lightning-gpt3.git'
# !curl https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml

import asyncio
import base64
import functools
import inspect
import io
import os
import time
from io import BytesIO
from typing import Any, Callable, Optional

import lightning as L
import torch
from ldm.lightning import LightningStableDiffusion, PromptDataset
from nicegui import ui
from pydantic import BaseModel

from lightning_gpt3 import LightningGPT3

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Text(BaseModel):
    text: Optional[str]


async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


def webpage(
    enhace_fn: Callable,
    predict_fn: Callable,
    host: str,
    port: int,
    reference_inference_time: Optional[float],
    reference_inference_time_prompt: Optional[float],
    source,
    enhance_prompt_text: str,
):
    async def progress_tracker():
        if progress.value >= 1.0 or progress.value == 0:
            return
        progress.value = round((progress.value * reference_inference_time + 0.1) / reference_inference_time, 2)

        # if progress_prompt.value >= 1.0 or progress_prompt.value == 0:
        #     return
        # progress_prompt.value = round(
        #     (progress_prompt.value * reference_inference_time_prompt + 0.1) / reference_inference_time_prompt, 3
        # )

    async def generate_image():
        nonlocal reference_inference_time
        t0 = time.time()
        progress.value = 0.0001
        image.source = "https://dummyimage.com/600x400/ccc/000000.png&text=building+image..."
        prediction = await io_bound(predict_fn, request=Text(text=enhance_prompt.value))
        image.source = prediction["image"]
        progress.value = 1.0
        reference_inference_time = time.time() - t0

    async def prompt_enhace():
        nonlocal reference_inference_time_prompt
        t0 = time.time()
        # progress_prompt.value = 0.0001
        prediction = await io_bound(enhace_fn, request=Text(text=prompt.value))
        enhance_prompt.value = prediction
        # progress_prompt.value = 1.0
        reference_inference_time_prompt = time.time() - t0

    # User Interface
    with ui.row().style("gap:10em"):
        with ui.column():
            ui.label("Stable Diffusion with Prompt Enhancement with Lightning AI").classes("text-2xl")
            prompt = ui.input("Prompt").style("width: 50em")
            prompt.value = "a forest-inspired office space"
            # progress_prompt = ui.linear_progress()
            ui.timer(interval=0.1, callback=progress_tracker)
            ui.button("Prompt Enhace", on_click=prompt_enhace).style("width: 15em")

            enhance_prompt = ui.input("Edit Enhanced Prompt", on_change=lambda e: input_result.set_text(e.value)).style(
                "width: 50em"
            )
            ui.label("Enhanced Prompt:").style("font-size: 110%")
            input_result = ui.label().style("width: 50em")
            enhance_prompt.value = enhance_prompt_text
            ui.button("Generate", on_click=generate_image).style("width: 15em")
            progress = ui.linear_progress()
            ui.timer(interval=0.1, callback=progress_tracker)
        image = ui.image().style("width: 60em")
        image.source = source

    # Note: Hack to enable running in spawn context.
    def stack_patch():
        class FakeFrame:
            filename = "random"

        return [FakeFrame(), None]

    inspect.stack = stack_patch
    ui.run(host=host, port=port, reload=False)


class DiffusionServeInteractive(L.LightningWork):
    _start_method = "spawn"

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
        self._gpt3 = LightningGPT3(api_key=os.getenv("OPENAI_API_KEY"))

    def prompt_enhace(self, request):
        prompt = "Describe a " + request.text + " picture"
        enhanced_prompt = self._gpt3.generate(prompt=prompt, max_tokens=80)[2::]
        return enhanced_prompt

    def predict(self, request):
        with torch.no_grad():
            image = self._trainer.predict(self._model, torch.utils.data.DataLoader(PromptDataset([request.text])))[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

    def run(self):
        self.setup()
        t0 = time.time()
        prompt = self.prompt_enhace(Text(text="a forest-inpired office space"))
        t1 = time.time()
        image = self.predict(Text(text=prompt))["image"]
        t2 = time.time()
        print(t2 - t1)
        webpage(
            self.prompt_enhace,
            self.predict,
            self.host,
            self.port,
            t2 - t1,
            t0 - t1,
            image,
            prompt,
        )


component = DiffusionServeInteractive(cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80))

app = L.LightningApp(component)
