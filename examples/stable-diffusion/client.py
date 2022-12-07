import base64, io, requests, PIL.Image as Image

if __name__ == "__main__":
    response = requests.post(
        "https://tiuvn-01gkqb1e6zfj2gh3aqz2kvvcx4.litng-ai-03.litng.ai/predict",
        json={"text": "forest-inspired bedroom"},
    )
    print
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
