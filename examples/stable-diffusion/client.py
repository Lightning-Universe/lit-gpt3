import base64
import io

import PIL.Image as Image
import requests

if __name__ == "__main__":
    response = requests.post(
        "http://127.0.0.1:7777/predict",
        json={"text": "Harry potter-inspired bedroom"},
    )
    print
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
