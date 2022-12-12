import base64, io, requests, PIL.Image as Image

if __name__ == "__main__":
    response = requests.post(
<<<<<<< HEAD
        "https://aeuym-01gm3xcetgn22b54g4m1ntafew.litng-ai-03.litng.ai/predict",
        json={"text": "firework-inspired bedroom"},
    )

=======
        "https://wxoak-01gkptph8n1ztya1rt64ze4a30.litng-ai-03.litng.ai/predict",
        json={"text": "Harry potter-inspired bedroom"},
    )
    print
>>>>>>> main
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["image"][22:])))
    image.save("response.png")
