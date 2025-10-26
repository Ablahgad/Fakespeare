from openai import OpenAI
import base64
import os
import wave
import click
import re

BOSON_API_KEY = os.getenv("BOSON_API_KEY")

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

# @click.command()
# @click.option(
#     "--transcript",
#     type=str,
#     default= r"C:\Users\ablah\repos\SAS\TestingMultitalk\tomorrow.txt",
#     help="The prompt to use for generation. If not set, we will use a default prompt.",
# )
# @click.option(
#     "--scene_prompt",
#     type=str,
#     default=f"./quiet_indoor.txt",
#     help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
# )
def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

def generate_prompt_scene_description(scene_prompt, scene_prompt_given):
    """ Generate scene description block for system message.
    Args:
        scene_prompt (str): The scene description prompt.
        scene_prompt_given (bool): Whether a scene prompt was provided.
    """
    if scene_prompt_given:
        return f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
    return ""

def extract_scene_description(full_transcript):
    """ Extract scene description from transcript if present.
    Args:
        full_transcript (str): The full transcript text.

    Returns:
        scene_prompt (str): The extracted scene description.
        scene_prompt_given (bool): Whether a scene description was found.
        remaining_transcript (str): The transcript without the scene description.
    """
    lines = full_transcript.split("\n")

    setting_index = lines.index("SETTING:") if "SETTING:" in lines else -1
    
    # if no "SETTING:" found
    if setting_index == -1:
        return "", False, full_transcript
    
    # if "SETTING:" found, find the end of the setting block (can be multiple lines)
    for i in range(setting_index + 1, len(lines)):
        if lines[i] == "":
            setting_end_index = i
            break
    else:
        setting_end_index = len(lines)

    # Extract the scene description and remaining transcript
    scene_prompt = "\n".join(lines[setting_index + 1:setting_end_index]).strip()
    remaining_transcript = "\n".join(lines[setting_end_index:]).strip()
    return scene_prompt, True, remaining_transcript


"""
process:

- Read transcript from file
- Clean transcript (remove unwanted characters, replace tags) DONE
- Extract scene description if present DONE
- Construct system scene description with the tags DONE
"""

def formated_script(script):
    script = script.replace("(", " ")
    script = script.replace(")", " ")
    script = script.replace("°F", " degrees Fahrenheit")
    script = script.replace("°C", " degrees Celsius")

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
        script = script.replace(tag, replacement)

    script = script.replace("[", "<|speaker_id_start|>")
    script = script.replace("]", "<|speaker_id_end|>")
    lines = script.split("\n")

    script = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    script = script.strip()
    return script

def extract_dialogue(script, actor_speaker):
    """ Extract dialogue turns from the script.
    Args:
        script (str): The full script text. (excluding scene description)
        actor_speaker (dictionary): A dictionary mapping actor names to their speaker tags. """
    
    lines = script.split("\n")

    dialogue = []

    for i in range(len(lines)):
        if lines[i] in actor_speaker.keys():
            for j in range(i+1, len(lines)):
                if lines[j] in actor_speaker.keys() or lines[j] == "":
                    break
                cur_tag = actor_speaker[lines[i]]
                cur_speech = f"[{cur_tag}]{lines[j]}"
                dialogue.append(cur_speech)
    
    return "\n".join(dialogue)
            

# do a dictionary mapping actor name to voice description
# and another mapping speaker tag to actor name (opposite)

def actor_speaker_mapping(actor_voice_desc):
    actor_speaker = {}
    for actor in actor_voice_desc.keys():
        speaker_tag = f"SPEAKER{len(actor_speaker)+1}"
        actor_speaker[actor] = speaker_tag
    return actor_speaker



def main():
    if os.path.exists(r"sample_ft.txt"):
        with open(r"sample_ft.txt", "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    transcript = formated_script(transcript)

    scene_prompt, scene_prompt_given, remaining_transcript = extract_scene_description(transcript)

    final_scene_prompt = generate_prompt_scene_description(scene_prompt, scene_prompt_given)

    actor_speaker_tags = actor_speaker_mapping({"Macbeth": "masculine voice", "Lady Macbeth": "feminine voice"})

    dialogue = extract_dialogue(remaining_transcript, actor_speaker_tags)

    client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

    final_content = "Generate realistic multi-speaker audio." + final_scene_prompt + "\n" + dialogue

    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            
            {"role": "system", "content": "You are an AI assistant designed to convert text into speech. If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice. If no speaker tag is present, select a suitable voice on your own."},
            {"role": "user", "content": transcript},
            {"role": "system", "content": final_content}
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
    open("gen6.wav", "wb").write(base64.b64decode(audio_b64))

if __name__ == "__main__":
    main()
