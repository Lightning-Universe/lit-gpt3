import base64
import io

import PIL.Image as Image
import requests

if __name__ == "__main__":
    response = requests.post(
        "https://siqzv-01gpkvqpy5k6g7sw5s9nckvhnr.litng-ai-03.litng.ai/predict",
        json={"text": "forest-inspired bedroom", "enhacement": True},
    )

    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
    print(response.json()["Enhanced_prompt"])
