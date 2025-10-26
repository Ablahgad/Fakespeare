import click
import soundfile as sf
import langid
import jieba
import os # need
import re
import copy
import torchaudio
import tqdm
# import yaml
from openai import OpenAI #need
import os #need
import base64 #need
import shutil
import wave
import json


# from loguru import logger # for logging what is going on for debugging

from typing import List
# from transformers import AutoConfig, AutoTokenizer # don't need, local instance
# from transformers.cache_utils import StaticCache # gives us errors for some reason
from typing import Optional
from dataclasses import asdict
import torch

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


"""
possible add a function to "chunk" over here... maybe modified from their script

OG: prepare_chunk_text

If we don't do chunking, just have [text] instead of the chunk function call
"""
"""
def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret
"""

# We don't need the class HiggsAudioModelClient, it is only for local inference


def generate_reference_audio_from_description(client, character_name, voice_description, output_path):
    prompt = f"[{character_name}] {voice_description}"
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )

    audio_b64 = response.choices[0].message.audio.data
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

def prepare_generation_context_api(
    client,
    scene_prompt: str,
    reference_map: dict,
    dialogue_text: str,
    ref_audio_dir: str = "./ref_audio"
) -> list:
    """
    Prepares OpenAI-style messages for Boson API using reference audio and transcripts.
    Automatically generates reference audio from voice description if missing.

    Args:
        client: OpenAI-compatible Boson API client.
        scene_prompt (str): Scene description to embed.
        reference_map (dict): Maps speaker tags to dicts with 'transcript', 'audio_path' (optional), and 'voice_description'.
        dialogue_text (str): The actual dialogue with speaker tags.
        ref_audio_dir (str): Directory to save generated reference audio files.

    Returns:
        List[dict]: Messages formatted for chat.completions.create
    """
    messages = []

    # Scene description
    if scene_prompt:
        messages.append({
            "role": "system",
            "content": f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
        })

    # Speaker tag guidance
    messages.append({
        "role": "system",
        "content": (
            "If the user's message includes a [SPEAKER] tag, do not read the tag aloud. "
            "Instead, use the corresponding reference audio and transcript to condition the voice. "
            "If no speaker tag is present, select a suitable voice automatically."
        )
    })

    speaker_descriptions = []
    for speaker, ref in reference_map.items():
        desc = ref.get("voice_description", "a unique voice")
        speaker_descriptions.append(
            f"<|speaker_id_start|>{speaker}<|speaker_id_end|> {desc}"
        )

    if speaker_descriptions:
        messages.append({
            "role": "system",
            "content": "\n".join(speaker_descriptions)
        })


    # Reference audio + transcript for each speaker
    for speaker, ref in reference_map.items():
        transcript = ref.get("transcript") or ref.get("voice_description")
        audio_path = ref.get("audio_path")

        # If audio is missing, generate it from voice description
        if not audio_path:
            audio_path = os.path.join(ref_audio_dir, f"{speaker}.wav")
            if not os.path.exists(audio_path):
                generate_reference_audio_from_description(client, speaker, ref["voice_description"], audio_path)
        else:
            # If audio_path is provided but not named as speaker.wav, copy it
            target_path = os.path.join(ref_audio_dir, f"{speaker}.wav")
            if os.path.abspath(audio_path) != os.path.abspath(target_path):
                if not os.path.exists(target_path):
                    shutil.copy(audio_path, target_path)
            audio_path = target_path  # Use the renamed copy

        messages.append({
            "role": "user",
            "content": f"[{speaker}] {transcript}"
        })
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": b64(audio_path),
                    "format": "wav"
                }
            }]
        })


    messages.append({
    "role": "system",
    "content": (
        "The following user message contains multiple speakers. "
        "Use the reference audio and transcript provided earlier to condition each speaker's voice. "
        "Do not read the speaker tags aloud."
    )
    })


    # Final dialogue to generate
    messages.append({
        "role": "user",
        "content": dialogue_text
    })

    return messages

@click.command()
@click.option(
    "--transcript",
    type=str,
    default= "fight.txt",
    help="The prompt to use for generation. If not set, we will use a default prompt.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"{CURR_DIR}/scene_prompt/heavy_rain.txt",
    help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
)

# if we do chunking, we can add these options


# @click.option(
#     "--chunk_method",
#     default=None,
#     type=click.Choice([None, "speaker", "word"]),
#     help="The method to use for chunking the prompt text. Options are 'speaker', 'word', or None. By default, we won't use any chunking and will feed the whole text to the model.",
# )
# @click.option(
#     "--chunk_max_word_num",
#     default=200,
#     type=int,
#     help="The maximum number of words for each chunk when 'word' chunking method is used. Only used when --chunk_method is set to 'word'.",
# )
# @click.option(
#     "--chunk_max_num_turns",
#     default=1,
#     type=int,
#     help="The maximum number of turns for each chunk when 'speaker' chunking method is used. Only used when --chunk_method is set to 'speaker'.",
# )


def main(transcript, scene_prompt, ref_audio_dir = "./ref_audio"):
    # Load Boson API client
    BOSON_API_KEY = os.getenv("BOSON_API_KEY")
    client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

    # Load dialogue text
    if os.path.exists(r"TestingMultitalk\tomorrow.txt"):
        with open(r"TestingMultitalk\tomorrow.txt", "r", encoding="utf-8") as f:
            transcript = f.read().strip()

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

    # Load scene prompt
    if scene_prompt.lower() == "":
        scene_text = ""
    else:
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_text = f.read().strip()

    # Define speaker reference map manually or load from config
    reference_map = {
        "SPEAKER1": {
            "voice_description": "A warm, thoughtful woman with a soft British accent."
        },
        "SPEAKER2": {
            "audio_path": os.path.join(ref_audio_dir, "shrek_donkey.wav"),
            "transcript": "And I've got a great idea, I'll stick with you. You're a mean green fighting machine, together we'll scare the spit out of anybody that crosses us. \
Oh, Wow, that was really scary. And if you don't mind me saying, if that don't work, your breath certainly will get the job done, 'cause you definitely need some Tic Tacs or something, 'cause your breath stinks!"
        }
    }

    # Prepare messages
    messages = prepare_generation_context_api(
        client=client,
        scene_prompt=scene_text,
        reference_map=reference_map,
        dialogue_text=transcript,
        ref_audio_dir=ref_audio_dir
    )

    # Call Boson API
    response = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=0.8,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )

    # print(json.dumps(messages, indent=2))
    
    # Save audio
    audio_b64 = response.choices[0].message.audio.data
    open("gen2_out.wav", "wb").write(base64.b64decode(audio_b64))

    print("Audio saved")

if __name__ == "__main__":
    main()