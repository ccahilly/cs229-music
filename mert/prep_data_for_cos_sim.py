import os
import numpy as np
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor
import torch
import torchaudio.transforms as T
from random import shuffle
import os
import torchaudio.functional as F


data_dir = "../data/mert"
wav_dir = "../data/wav"
num_songs = 10
original_sampling_rate = 16000

os.makedirs(data_dir, exist_ok=True)

def main():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # Sampling rate for the model processor
    desired_sampling_rate = processor.sampling_rate

    # List all .wav files in the directory
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]

    # Shuffle and pick 100 random files
    shuffle(wav_files)
    wav_files = wav_files[:num_songs]

    # Initialize resampler and processor
    resampler = T.Resample(orig_freq=original_sampling_rate, new_freq=desired_sampling_rate)

    # Prepare dataset
    data = []

    # Process each audio file
    for file in wav_files:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file)

        # Resample if the sample rate doesn't match the desired one
        if sample_rate != desired_sampling_rate and sample_rate == original_sampling_rate:
            waveform = resampler(waveform)
        
        # Convert stereo to mono if necessary
        if waveform.ndim == 2:  # Stereo audio
            waveform = waveform.mean(axis=0)  # Average the two channels
        
        # Normalize the waveform to match input expected by the model
        audio_array = waveform.squeeze().numpy()
        
        # Add to the dataset list
        data.append({
            "audio_array": audio_array,
            "audio_path": file,
            "sampling_rate": desired_sampling_rate
        })

    # Create a Hugging Face dataset
    dataset = Dataset.from_dict({"audio": data})

    # Save the dataset to disk in the expected format
    dataset.save_to_disk("../data/mert")

    # Print the dataset to verify
    print(dataset)

if __name__ == "__main__":
    main()
