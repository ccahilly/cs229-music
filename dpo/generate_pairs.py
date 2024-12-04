import pandas as pd
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import os
from scipy.io.wavfile import write
import numpy as np

# Constants
iteration_number = 0

NUM_CAPTIONS = 1  # Set the number of captions to select
NUM_PAIRS_PER_CAPTION_PER_TEMP = 1  # Set the number of pairs per caption
TEMPS = [0.7, 1.0]

audio_dir = f"../data/dpo-gen-{iteration_number}/wavs"
logprobs_dir = f"../data/dpo-gen-{iteration_number}/logprobs"
caption_file = "../data/musiccaps-train-data.csv"  # Path to the captions CSV

FAILED_YTID_PATH = "../data/failed_ytids.txt"  # Path to failed ytids

SAMPLE_RATE = 32000  # Sample rate for the audio clips
COMPRESSION_RATIO = 50
SECONDS = 10

# To start, the reference model and policy model are the same
reference_model_name = "facebook/musicgen-small"
policy_model_name = "facebook/musicgen-small"

REF_IDX = 0
POL_IDX = 1

def logits_to_logprobs(logits):
    # Assuming logits are of shape (batch_size = 1, num_classes)
    probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    logprobs = torch.log(probs)  # Take log of the probabilities
    return logprobs

# Function to generate 10 second audio for a caption
def generate_audio_for_caption(caption, ytid, processor_name, model, device, num_pairs=NUM_PAIRS_PER_CAPTION_PER_TEMP, temps=TEMPS):
    # Load the processor
    if processor_name == "reference":
        processor = AutoProcessor.from_pretrained(reference_model_name)
        idx = REF_IDX
    elif processor_name == "policy":
        processor = AutoProcessor.from_pretrained(policy_model_name)
        idx = POL_IDX
    else:
        raise ValueError(f"Invalid processor name: {processor_name}")
    
    # Prepare the prompt for MusicGen
    inputs = processor(text=caption, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate the required number of pairs
    print(f"\nGenerating audio for caption: {caption}\n")
    for i in range(num_pairs):
        for temp in temps:
            # Generate music using MusicGen
            generated_audio = model.generate(**inputs, max_new_tokens=COMPRESSION_RATIO * SECONDS, temperature=temp)

            # Extract logits (model output before softmax)
            logit = generated_audio.logit()
            print(f"Logit: {logit}")

            # Convert output tokens back to audio
            audio_array = generated_audio.cpu().numpy()

            # Save the audio as a WAV file
            output_path = os.path.join(audio_dir, f"{ytid}-temp{temp}-pair{i}-{idx}.wav")
            write(output_path, SAMPLE_RATE, audio_array)

            prob = torch.softmax(logit, dim=-1)
            logprob = torch.log(prob)
            print(f"Logprobs: {logprob}")
            print(f"Logprobs: {logprob.shape}")

            # Save logits as a numpy array
            logprobs_path = os.path.join(logprobs_dir, f"{ytid}-temp{temp}-pair{i}-{idx}.npy")
            np.save(logprobs_path, logprob.cpu().numpy())

            print(f"Saved as {output_path}")

def gen(model, processor_name, sampled_data):
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    model.to(device)
    print(f"Device: {device}")

    # Generate audio for each sampled caption
    for _, row in sampled_data.iterrows():
        caption = row['caption']
        ytid = row['ytid']
        
        # Generate pairs of audio for the caption
        generate_audio_for_caption(caption, ytid, processor_name, model, device)

def main():
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(logprobs_dir, exist_ok=True)

    # Load the failed ytids
    with open(FAILED_YTID_PATH, "r") as f:
        failed_ytids = set(f.read().splitlines())

    # Load the captions dataset
    data = pd.read_csv(caption_file)

    # Filter captions based on failed ytids
    filtered_data = data[data["ytid"].isin(failed_ytids)]

    # Sample NUM_CAPTIONS captions randomly
    sampled_data = filtered_data.sample(NUM_CAPTIONS, random_state=42)

    # Load the pre-trained MusicGen model and processor
    reference_processor = AutoProcessor.from_pretrained(reference_model_name)
    reference_model = MusicgenForConditionalGeneration.from_pretrained(reference_model_name)
    
    policy_processor = AutoProcessor.from_pretrained(policy_model_name)
    policy_model = MusicgenForConditionalGeneration.from_pretrained(policy_model_name)

    # Generate audio for the sampled captions
    gen(reference_model, "reference", sampled_data)
    gen(policy_model, "policy", sampled_data)

if __name__ == "__main__":
    main()