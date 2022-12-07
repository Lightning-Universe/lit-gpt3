import openai
import warnings

GPT_MODEL = "text-davinci-003"


class LightningGPT3:
    """This `LightningFlow` component allows integrating GPT-3 into your Lightning App.

    The `generate()` method can be used to generate text from a prompt. The `run()` method is a no-op, since this
    component doesn't need to run anything.
    """

    def __init__(self, api_key: str):
        super().__init__()

        openai.api_key = api_key
        try:
            openai.Model.list()
        except:
            raise Exception("Sorry, you provided an invalid API Key")
            
    def generate(self, prompt: str, max_tokens: int = 20):
        if max_tokens<15:
            warnings.warn('Warning Message: the max_token variable is too small, your prompts may lack iformation, try max_tokens>=15')
            
        response = openai.Completion.create(model=GPT_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        return response["choices"][0]["text"]

    def run(self):
        # NOTE: This component just acts as a wrapper around the OpenAI API, so we don't need to run anything.
        pass
