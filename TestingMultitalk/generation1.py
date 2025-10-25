import openai
import os
import wave
import click

BOSON_API_KEY = os.getenv("BOSON_API_KEY")

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

@click.command()
@click.option(
    "--transcript",
    type=str,
    default= r"C:\Users\ablah\repos\higgs-audio\TestingMultitalk\en_argument.txt",
    help="The prompt to use for generation. If not set, we will use a default prompt.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"./quiet_indoor.txt",
    help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
)

def main(transcript, scene_prompt):
    if os.path.exists(transcript):
        with open(transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    # if len(speaker_tags) > 1:
    #     # By default, we just alternate between male and female voices
    #     speaker_desc_l = []

    #     for idx, tag in enumerate(speaker_tags):
    #         if idx % 2 == 0:
    #             speaker_desc = f"feminine"
    #         else:
    #             speaker_desc = f"masculine"
    #         speaker_desc_l.append(f"{tag}: {speaker_desc}")
    #     #This is where we add speaker personalities and traits

    #     speaker_desc = "\n".join(speaker_desc_l)
    #     scene_desc_l = []
    #     if scene_prompt:
    #         scene_desc_l.append(scene_prompt)
    #     scene_desc_l.append(speaker_desc)
    #     scene_desc = "\n\n".join(scene_desc_l)

    #     # system_message = Message(
    #     #     role="system",
    #     #     content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
    #     # )
    # else:
    #     system_message_l = ["Generate audio following instruction."]
    #     if scene_prompt:
    #         system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
    #     # system_message = Message(
    #     #     role="system",
    #     #     content="\n\n".join(system_message_l),
    #     # )
        
    # pattern = re.compile(r"\[(SPEAKER\d+)\]")

    # speaker_tags = sorted(set(pattern.findall(transcript)))
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

    client = openai.Client(
        api_key=BOSON_API_KEY,
        base_url="https://hackathon.boson.ai/v1"
    )

    # for this api, we onlu support PCM format output
    response = client.audio.speech.create(
        model="higgs-audio-generation-Hackathon",
        voice="belinda",
        input=transcript,
        response_format="pcm"
    )

    # You can use these parameters to write PCM data to a WAV file
    num_channels = 1        
    sample_width = 2        
    sample_rate = 24000   

    pcm_data = response.content

    with wave.open('belinda_test.wav', 'wb') as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)

if __name__ == "__main__":
    main()
