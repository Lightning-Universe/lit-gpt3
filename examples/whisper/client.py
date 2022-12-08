import requests, base64

f = open("dani.mp3", "rb")
waveform = base64.b64encode(f.read()).decode("utf-8")
f.close()

response = requests.post(
    "https://dslma-01gkqgdn2gzv9fm1v43p1dxer8.litng-ai-03.litng.ai/predict", json={"text": waveform}
)
print(response.json()["text"])
