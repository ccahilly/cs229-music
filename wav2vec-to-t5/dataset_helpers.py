NORMALIZING_INPUT = True  # Flag for normalization
MAX_TOKENS = 64
BATCH_SIZE = 8

from scipy.io import wavfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def preprocess_audio(audio_path):
    """
    Preprocess audio file to ensure it is mono and normalized.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        np.ndarray: Preprocessed audio data.
    """
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # Convert stereo to mono if necessary
    if audio.ndim == 2:  # Stereo audio
        audio = audio.mean(axis=1)  # Average the two channels

    # Normalize audio to the range [-1, 1] if required
    if NORMALIZING_INPUT:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    return audio, sample_rate

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
        processed_audio, sample_rate = preprocess_audio(audio_path)
        if sample_rate != 16000:
            raise ValueError(f"Invalid sample rate: {sample_rate}. Expected 16000 Hz.")
        
        inputs = self.processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt")

        # Tokenize caption
        labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS)

        # Check if attention_mask is present
        input_values = inputs["input_values"].squeeze(0)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_values))  # Default to ones if missing

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels["input_ids"].squeeze(0),
            "decoder_attention_mask": labels["attention_mask"].squeeze(0)
        }