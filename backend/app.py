from openai import OpenAI
import base64
import os
import wave
import click
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BOSON_API_KEY = os.getenv("BOSON_API_KEY")

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

@app.route("/generate_audio", methods=["POST"])
def main():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    uploaded_file = request.files['file']

    transcript = uploaded_file.read().decode('utf-8')

    if not transcript:
        return jsonify({"error": "No text provided"}), 400   

    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)

    transcript = transcript.replace("[", "<|speaker_id_start|>")
    transcript = transcript.replace("]", "<|speaker_id_end|>")
    lines = transcript.split("\n")

    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()
    print(transcript)

    BOSON_API_KEY = os.getenv("BOSON_API_KEY")
    client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            
            {"role": "system", "content": "You are an AI assistant designed to convert text into speech. If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice. If no speaker tag is present, select a suitable voice on your own."},
            {"role": "user", "content": transcript},
            {"role": "system", "content": f"""Generate realistic multi-speaker audio.
             
                <|scene_desc_start|>
                It is raining heavily, and thunder rolls in the distance.
                <|scene_desc_end|>

                <|speaker_id_start|>SPEAKER1<|speaker_id_end|> masculine voice
                <|speaker_id_start|>SPEAKER2<|speaker_id_end|> feminine voice
                """}
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )

    audio_b64 = resp.choices[0].message.audio.data
    open("output.wav", "wb").write(base64.b64decode(audio_b64))

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    return send_file(tmp_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
