import os

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

    print(f"Full transcript lines: {lines}")

    setting_index = lines.index("SETTING:") if "SETTING:" in lines else -1

    print(f"SETTING index: {setting_index}")
    
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

if __name__ == "__main__":
    full_transcript = "sample_ft.txt"
    if os.path.exists(full_transcript):
        with open(full_transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    scene_prompt, scene_prompt_given, remaining_transcript = extract_scene_description(transcript)
    print("\nExtracted Scene Description:")
    print(scene_prompt)
    print("\nRemaining Transcript:")
    print(remaining_transcript)

    actor_speaker = {
        "Macbeth": "SPEAKER1",
        "Lady Macbeth": "SPEAKER2"
    }
    dialogue_transcript = extract_dialogue(remaining_transcript, actor_speaker)
    print("\nExtracted Dialogue Transcript:")
    print(dialogue_transcript)