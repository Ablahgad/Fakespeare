import os
import base64
import torch
import torchaudio
from openai import OpenAI

# Setup
BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

def generate_reference_audio_from_description(client, speaker, voice_description, output_path):
    prompt = f"[{speaker}] {voice_description}"
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )
    audio_b64 = response.choices[0].message.audio.data
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

def generate_audio(text, speaker_tag, reference_audio_path, transcript, output_path):
    messages = [
        {"role": "user", "content": f"[{speaker_tag}] {transcript}"},
        {"role": "assistant", "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(open(reference_audio_path, "rb").read()).decode("utf-8"),
                "format": "wav"
            }
        }]},
        {"role": "user", "content": text}
    ]
    response = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>", "[SPEAKER1]", "[SPEAKER2]"],
        extra_body={"top_k": 50},
    )
    audio_b64 = response.choices[0].message.audio.data
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

def parse_dialogue(dialogue_path):
    with open(dialogue_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    turns = []
    current_speaker = None
    buffer = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current_speaker and buffer:
                turns.append((current_speaker, " ".join(buffer)))
                buffer = []
            current_speaker = line.strip("[]")
        else:
            buffer.append(line)
    if current_speaker and buffer:
        turns.append((current_speaker, " ".join(buffer)))
    return turns

def split_waveform(waveform, num_chunks):
    total_frames = waveform.shape[1]
    chunk_size = total_frames // num_chunks
    chunks = [waveform[:, i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks

def main():
    ref_audio_dir = "./ref_audio"
    os.makedirs(ref_audio_dir, exist_ok=True)

    # Paths
    speaker1_txt = "speaker1.txt"
    speaker2_txt = "speaker2.txt"
    dialogue_txt = "fight.txt"
    speaker1_ref = os.path.join(ref_audio_dir, "speaker1.wav")
    speaker2_ref = os.path.join(ref_audio_dir, "speaker2.wav")

    # Generate reference audio if missing
    if not os.path.exists(speaker1_ref):
        generate_reference_audio_from_description(client, "SPEAKER1", "A warm, thoughtful woman with a soft British accent.", speaker1_ref)
    if not os.path.exists(speaker2_ref):
        generate_reference_audio_from_description(client, "SPEAKER2", "A confident, energetic man with a New York accent.", speaker2_ref)

    # Load speaker texts
    speaker1_lines = open(speaker1_txt, "r", encoding="utf-8").read().strip()
    speaker2_lines = open(speaker2_txt, "r", encoding="utf-8").read().strip()

    # Generate audio
    generate_audio(speaker1_lines, "SPEAKER1", speaker1_ref, "Sample line from SPEAKER1", "speaker1_output.wav")
    generate_audio(speaker2_lines, "SPEAKER2", speaker2_ref, "Sample line from SPEAKER2", "speaker2_output.wav")

    # Parse dialogue turns
    turns = parse_dialogue(dialogue_txt)
    s1_count = sum(1 for t in turns if t[0] == "SPEAKER1")
    s2_count = sum(1 for t in turns if t[0] == "SPEAKER2")

    # Load and split audio
    s1_waveform, sr = torchaudio.load("speaker1_output.wav")
    s2_waveform, _ = torchaudio.load("speaker2_output.wav")
    s1_chunks = split_waveform(s1_waveform, s1_count)
    s2_chunks = split_waveform(s2_waveform, s2_count)

    # Stitch audio
    stitched = []
    s1_idx, s2_idx = 0, 0
    for speaker, _ in turns:
        if speaker == "SPEAKER1":
            stitched.append(s1_chunks[s1_idx])
            s1_idx += 1
        elif speaker == "SPEAKER2":
            stitched.append(s2_chunks[s2_idx])
            s2_idx += 1

    final_waveform = torch.cat(stitched, dim=1)
    torchaudio.save("gen2_out.wav", final_waveform, sr)
    print("Final stitched audio saved to gen2_out.wav")

if __name__ == "__main__":
    main()