"""Example script for generating audio using HiggsAudio."""

import click
# import soundfile as sf
import langid
import jieba
import os
import re
import yaml
from openai import OpenAI
import os
import wave
from data_types import AudioContent, TextContent, Message, ChatMLSample

from typing import List
from typing import Optional
from dataclasses import asdict
import torch
import base64

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"


MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", or None. By default, we won't use any chunking and
        will feed the whole text to the model.
    replace_speaker_tag_with_special_tags : bool, optional
        Whether to replace speaker tags with special tokens, by default False
        If the flag is set to True, we will replace [SPEAKER0] with <|speaker_id_start|>SPEAKER0<|speaker_id_end|>
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk when "word" chunking method is used, by default 100
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used,

    Returns
    -------
    List[str]
        The list of text chunks.

    """
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
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


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

def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, speaker_tags):
    """Prepare the context for generation.

    The context contains the system message, user message, assistant message, and audio prompt if any.
    """
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            # By default, we just alternate between male and female voices
            speaker_desc_l = []

            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


@click.command()
@click.option(
    "--audio_tokenizer",
    type=str,
    default="bosonai/higgs-audio-v2-tokenizer",
    help="Audio tokenizer path, if not set, use the default one.",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="The maximum number of new tokens to generate.",
)
@click.option(
    "--transcript",
    type=str,
    default= r"TestingMultitalk\en_argument.txt",
    help="The prompt to use for generation. If not set, we will use a default prompt.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt",
    help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="The value used to module the next token probabilities.",
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
)
@click.option(
    "--top_p",
    type=float,
    default=0.95,
    help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
)
@click.option(
    "--ras_win_len",
    type=int,
    default=7,
    help="The window length for RAS sampling. If set to 0 or a negative value, we won't use RAS sampling.",
)
@click.option(
    "--ras_win_max_num_repeat",
    type=int,
    default=2,
    help="The maximum number of times to repeat the RAS window. Only used when --ras_win_len is set.",
)
@click.option(
    "--ref_audio",
    type=str,
    default=None,
    help="The voice prompt to use for generation. If not set, we will let the model randomly pick a voice. "
    "For multi-speaker generation, you can specify the prompts as `belinda,chadwick` and we will use the voice of belinda as SPEAKER0 and the voice of chadwick as SPEAKER1.",
)
@click.option(
    "--ref_audio_in_system_message",
    is_flag=True,
    default=False,
    help="Whether to include the voice prompt description in the system message.",
    show_default=True,
)
@click.option(
    "--chunk_method",
    default=None,
    type=click.Choice([None, "speaker", "word"]),
    help="The method to use for chunking the prompt text. Options are 'speaker', 'word', or None. By default, we won't use any chunking and will feed the whole text to the model.",
)
@click.option(
    "--chunk_max_word_num",
    default=200,
    type=int,
    help="The maximum number of words for each chunk when 'word' chunking method is used. Only used when --chunk_method is set to 'word'.",
)
@click.option(
    "--chunk_max_num_turns",
    default=1,
    type=int,
    help="The maximum number of turns for each chunk when 'speaker' chunking method is used. Only used when --chunk_method is set to 'speaker'.",
)
@click.option(
    "--generation_chunk_buffer_size",
    default=None,
    type=int,
    help="The maximal number of chunks to keep in the buffer. We will always keep the reference audios, and keep `max_chunk_buffer` chunks of generated audio.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for generation.",
)
@click.option(
    "--device_id",
    type=int,
    default=None,
    help="The device to run the model on.",
)
@click.option(
    "--out_path",
    type=str,
    default="generation.wav",
)
@click.option(
    "--use_static_kv_cache",
    type=int,
    default=1,
    help="Whether to use static KV cache for faster generation. Only works when using GPU.",
)
def main(
    transcript,
    scene_prompt,
    ref_audio,
    ref_audio_in_system_message,
    chunk_method,
    chunk_max_word_num,
    chunk_max_num_turns,
):

    pattern = re.compile(r"\[(SPEAKER\d+)\]") # do we need this?

    if os.path.exists(transcript):
        with open(transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    if scene_prompt is not None and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt = f.read().strip()
    else:
        scene_prompt = None

    speaker_tags = sorted(set(pattern.findall(transcript)))
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
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
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=ref_audio,
        ref_audio_in_system_message=ref_audio_in_system_message,
        audio_tokenizer=audio_tokenizer,
        speaker_tags=speaker_tags,
    )
    chunked_text = prepare_chunk_text(
        transcript,
        chunk_method=chunk_method,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
    )

    # Call Boson model to generate audio
    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=False  # non-streaming, full audio output
    )

    # Extract audio from the response
    audio_base64 = resp.choices[0].message["audio"]["data"]
    audio_bytes = base64.b64decode(audio_base64)

    # Save audio to WAV file
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)


if __name__ == "__main__":
    main()
