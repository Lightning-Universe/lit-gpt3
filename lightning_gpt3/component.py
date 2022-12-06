import lightning as L
import openai

GPT_MODEL = "text-davinci-003"


class LightningGPT3(L.LightningFlow):
    """This `LightningFlow` component allows integrating GPT-3 into your Lightning App.

    The `generate()` method can be used to generate text from a prompt. The `run()` method is a no-op, since this
    component doesn't need to run anything.
    """

    def __init__(self, api_key: str):
        super().__init__()

        openai.api_key = api_key
        # TODO: make sure provided `api_key` and is valid

    def generate(self, prompt: str, max_tokens: int = 20):
        response = openai.Completion.create(model=GPT_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.7)

        return response["choices"][0]["text"]

    def run(self):
        # NOTE: This component just acts as a wrapper around the OpenAI API, so we don't need to run anything.
        pass
