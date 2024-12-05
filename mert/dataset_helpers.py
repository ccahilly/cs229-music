NORMALIZING_INPUT = True  # Flag for normalization
MAX_TOKENS = 64

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

def preprocess_audio(audio_path, processor):
    """
    Preprocess audio file to ensure it is mono and normalized.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        np.ndarray: Preprocessed audio data.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != processor.sampling_rate:
        # print(f"resampling from {sample_rate} to {processor.sampling_rate}")
        resampler = T.Resample(orig_freq=sample_rate, new_freq=processor.sampling_rate)
        waveform = resampler(waveform)

    # Convert stereo to mono if necessary
    if waveform.ndim == 2:  # Stereo audio
        waveform = waveform.mean(axis=0)  # Average the two channels

    # Normalize audio to the range [-1, 1] if required
    # if NORMALIZING_INPUT:
    #     waveform = waveform.astype(np.float32) / np.iinfo(np.int16).max

    waveform = waveform.squeeze().numpy()
    # print(f"Waveform type: {type(waveform)}, shape: {waveform.shape}")

    # print(f"mert {processor.sampling_rate}")
    return waveform, processor.sampling_rate

# Dataset class
class AudioCaptionDataset(Dataset):
    def __init__(self, data_path, processor, tokenizer):
        self.data = pd.read_csv(data_path)
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["file_path"]
        caption = row["caption"]

        # Load and preprocess audio
        processed_audio, sample_rate = preprocess_audio(audio_path, self.processor)
        # if sample_rate != self.processor.sampling_rate:
        #     print("Value error")
        #     print(sample_rate)

        #     sample_rate = self.processor.sampling_rate

        # print(f"Processed audio shape: {processed_audio.shape}")

        inputs = self.processor(processed_audio, sampling_rate = sample_rate, return_tensors="pt")
        input_values = torch.tensor(processed_audio)
        # print(input_values.shape)

        attention_mask = inputs.get("attention_mask", torch.ones_like(input_values))  # Default to ones if missing

        # Tokenize caption
        labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS)

        return {
            "inputs": input_values,
            "attention_mask": attention_mask,
            "labels": labels["input_ids"].squeeze(0),
            "decoder_attention_mask": labels["attention_mask"].squeeze(0)
        }