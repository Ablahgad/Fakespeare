# Fakespeare
A submission for the 2025 Boson AI Hackathon.
## About This App
We at Fakespeare believe that a good script is the backbone of a good performance. Listen to your script with the actors of your choice with the Text-to-Audio Generator. Upload a text file, specify multiple voice actors, describe their voices, and generate high-quality spoken audio.

## How It Works
After you upload a text file, the app scrapes and modifies the content into a format suitable for Boson AI's Higgs Audio Generatoin model. This model then generates the audio file using the descriptions you provided for each actor.

## Instructions for Uploaded Files
To ensure the best results, please follow these guidelines when uploading files:

File types: Only .txt files are supported.
Formatting: Use clear line breaks between paragraphs and dialogue for each character.
Actor cues: If your script involves multiple actors, label lines or paragraphs with actor names (e.g., ACTOR 1 or JOHN).
Text length: Keep individual sections concise for better voice quality. Very long paragraphs may be truncated.
Special characters: Avoid excessive symbols or formatting that may confuse the model.
Following these instructions ensures that the uploaded text is properly parsed and that each voice actor sounds natural in the final audio.

## Instructions for running the code
Run the file by running the following into the terminal, note that the BosonAI API key must be set as an environment variable:

```
pip install -r requirements.txt
python ./backend/App.py
```

Then hosting the website locally through a live server.
