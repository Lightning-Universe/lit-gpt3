import requests, base64, os, gtts


def create_mp3(text):
    myobj = gtts.gTTS(text=text, lang="en", slow=False)
    myobj.save("reponse.mp3")
    # Playing the converted file


f = open("dani.mp3", "rb")
waveform = base64.b64encode(f.read()).decode("utf-8")
f.close()

response = requests.post(
    "https://uodfm-01gkshzer9kt7mnedd7vqps84c.litng-ai-03.litng.ai/predict", json={"text": waveform}
)

print(response.json()["text"])
create_mp3(response.json()["text"])
