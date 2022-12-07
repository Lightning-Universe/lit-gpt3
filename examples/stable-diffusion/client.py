import base64
import io

import PIL.Image as Image
import requests

if __name__ == "__main__":
    response = requests.post(
        "https://wxoak-01gkptph8n1ztya1rt64ze4a30.litng-ai-03.litng.ai/predict",
        json={"text": "Harry potter-inspired bedroom"},
    )
    print
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
