import pandas as pd
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import os
from scipy.io.wavfile import write
import numpy as np

# Constants
NUM_CAPTIONS = 5  # Set the number of captions to select
NUM_PAIRS_PER_CAPTION_PER_TEMP = 2  # Set the number of pairs per caption
TEMPS = [0.7, 1.0]

OUTPUT_DIR = "../data/dpo-gen-output"  # Directory to save generated audio clips
FAILED_YTID_PATH = "../data/failed_ytids.txt"  # Path to failed ytids
CAPTIONS_CSV_PATH = "../data/musiccaps-train-data.csv"  # Path to the captions CSV

SAMPLE_RATE = 32000  # Sample rate for the audio clips
COMPRESSION_RATIO = 50
SECONDS = 10

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the failed ytids
    with open(FAILED_YTID_PATH, "r") as f:
        failed_ytids = set(f.read().splitlines())

    # Load the captions dataset
    data = pd.read_csv(CAPTIONS_CSV_PATH)

    # Filter captions based on failed ytids
    filtered_data = data[data["ytid"].isin(failed_ytids)]

    # Sample NUM_CAPTIONS captions randomly
    sampled_data = filtered_data.sample(NUM_CAPTIONS, random_state=42)

    # Load the pre-trained MusicGen model and processor
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    model.to(device)
    print(f"Device: {device}")

    # Function to generate 10 second audio for a caption
    def generate_audio_for_caption(caption, ytid, num_pairs=NUM_PAIRS_PER_CAPTION_PER_TEMP, temps=TEMPS):
        # Prepare the prompt for MusicGen
        inputs = processor(text=caption, return_tensors="pt", truncation=True, padding=True).to(device)

        # Generate the required number of pairs
        print(f"\nGenerating audio for caption: {caption}\n")
        for i in range(num_pairs):
            for temp in temps:
                for j in range(2):
                    # Generate music using MusicGen
                    generated_audio = model.generate(**inputs, max_new_tokens=COMPRESSION_RATIO * SECONDS, temperature=temp)

                    # Convert output tokens back to audio
                    audio_array = generated_audio.cpu().numpy()

                    print(audio_array.shape)

                    # Save the audio as a WAV file
                    output_path = os.path.join(OUTPUT_DIR, f"{ytid}-temp{temp}-pair{i}-{j}.wav")
                    write(output_path, SAMPLE_RATE, audio_array)

                    print(f"Saved as {output_path}")

    # Generate audio for each sampled caption
    for _, row in sampled_data.iterrows():
        caption = row['caption']
        ytid = row['ytid']
        
        # Generate pairs of audio for the caption
        generate_audio_for_caption(caption, ytid)

if __name__ == "__main__":
    main()