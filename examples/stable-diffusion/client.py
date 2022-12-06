import base64
import io

import PIL.Image as Image
import requests

response = requests.post(
    "https://beutr-01gkmxry8pdqvetyf59xz8qrzv.litng-ai-03.litng.ai/predict",
    json={"text": "Harry potter-inspired bedroom"},
)
image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
image.save("response.png")
