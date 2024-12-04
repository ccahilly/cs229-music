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
import pandas as pd

data_dir = "../data/mert"

wav_dir = "../data/wav-48"
original_sampling_rate = 48000

train_metadata_path = "../data/splits/train.csv"
test_metadata_path = "../data/splits/val.csv"
val_metadata_path = "../data/splits/val.csv"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(data_dir + "/train", exist_ok=True)
os.makedirs(data_dir + "/test", exist_ok=True)
os.makedirs(data_dir + "/val", exist_ok=True)

def get_list_of_filepaths(metadata_path):
     df = pd.read_csv(metadata_path)
     return [os.path.join(wav_dir, f) for f in df["file_path"]]
     
def main():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    desired_sampling_rate = processor.sampling_rate

    for metadata_path in [train_metadata_path, test_metadata_path, val_metadata_path]:
        # List all .wav files in the directory
        wav_files = get_list_of_filepaths(metadata_path)

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