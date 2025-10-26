"""Example script for generating audio using HiggsAudio."""

import click
import os
import re
import yaml
import jieba
import langid
import base64
from typing import List, Optional
from dataclasses import asdict
from openai import OpenAI
from data_types import AudioContent, TextContent, Message

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")


def prepare_chunk_text(
    text: str,
    chunk_method: Optional[str] = None,
    chunk_max_word_num: int = 100,
    chunk_max_num_turns: int = 1,
) -> List[str]:
    """Chunk text by speaker or word count."""
    if chunk_method is None:
        return [text]

    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                speaker_utterance += ("\n" + line) if speaker_utterance else line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())

        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunks.append("\n".join(speaker_chunks[i : i + chunk_max_num_turns]))
            return merged_chunks

        return speaker_chunks

    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []

        for paragraph in paragraphs:
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunks.append("".join(words[i:i + chunk_max_word_num]))
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunks.append(" ".join(words[i:i + chunk_max_word_num]))

            chunks[-1] += "\n\n"
        return chunks

    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message: str) -> Message:
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]

    if system_message:
        contents.append(TextContent(system_message))

    return Message(role="system", content=contents)


def prepare_generation_context(
    scene_prompt: Optional[str],
    ref_audio: Optional[str],
    ref_audio_in_system_message: bool,
    speaker_tags: List[str],
    audio_tokenizer,
) -> tuple[list[Message], list]:
    """Prepare context for generation including system message and reference audio."""
    system_message = None
    messages = []
    audio_ids = []

    if ref_audio:
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([s.startswith("profile:") for s in speaker_info_l]):
            ref_audio_in_system_message = True

        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, name in enumerate(speaker_info_l):
                if name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][name[len("profile:"):].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")

            scene_text = f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>" if scene_prompt else "<|scene_desc_start|>\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
            system_message = _build_system_message_with_audio_prompt(f"Generate audio following instruction.\n\n{scene_text}")
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )

        for spk_id, name in enumerate(speaker_info_l):
            if not name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{name}.txt")
                assert os.path.exists(prompt_audio_path), f"Audio file {prompt_audio_path} not found."
                assert os.path.exists(prompt_text_path), f"Text file {prompt_text_path} not found."

                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()

                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(Message(role="user", content=f"[SPEAKER{spk_id}] {prompt_text}" if len(speaker_info_l) > 1 else prompt_text))
                    messages.append(Message(role="assistant", content=AudioContent(audio_url=prompt_audio_path)))
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = [f"{tag}: {'feminine' if i % 2 == 0 else 'masculine'}" for i, tag in enumerate(speaker_tags)]
            scene_desc = "\n\n".join([scene_prompt] + speaker_desc_l) if scene_prompt else "\n".join(speaker_desc_l)
            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            content = ["Generate audio following instruction."]
            if scene_prompt:
                content.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(role="system", content="\n\n".join(content))

    if system_message:
        messages.insert(0, system_message)

    return messages, audio_ids


@click.command()
@click.option("--transcript", type=str, default=r"TestingMultitalk\en_argument.txt")
@click.option("--scene_prompt", type=str, default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt")
@click.option("--ref_audio", type=str, default=None)
@click.option("--ref_audio_in_system_message", is_flag=True, default=False)
@click.option("--chunk_method", default=None, type=click.Choice([None, "speaker", "word"]))
@click.option("--chunk_max_word_num", default=200, type=int)
@click.option("--chunk_max_num_turns", default=1, type=int)
@click.option("--out_path", type=str, default="generation.wav")
def main(
    transcript,
    scene_prompt,
    ref_audio,
    ref_audio_in_system_message,
    chunk_method,
    chunk_max_word_num,
    chunk_max_num_turns,
    out_path,
):
    from transformers import AutoTokenizer

    # Load transcript
    if os.path.exists(transcript):
        with open(transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    # Load scene prompt
    if scene_prompt and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt = f.read().strip()
    else:
        scene_prompt = None

    # Speaker tags
    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    speaker_tags = sorted(set(pattern.findall(transcript)))

    # Clean transcript
    transcript = transcript.replace("(", " ").replace(")", " ").replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
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
    transcript = "\n".join([" ".join(line.split()) for line in transcript.split("\n") if line.strip()]).strip()
    if not any(transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
        transcript += "."

    # Load tokenizer
    audio_tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-tokenizer")

    # Prepare context and chunks
    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=ref_audio,
        ref_audio_in_system_message=ref_audio_in_system_message,
        speaker_tags=speaker_tags,
        audio_tokenizer=audio_tokenizer,
    )
    chunked_text = prepare_chunk_text(
        transcript,
        chunk_method=chunk_method,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
    )

    # Generate audio
    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=False
    )

    audio_base64 = resp.choices[0].message["audio"]["data"]
    audio_bytes = base64.b64decode(audio_base64)

    # Save audio
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    print(f"Audio saved to {out_path}")


if __name__ == "__main__":
    main()
