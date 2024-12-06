NORMALIZING_INPUT = True  # Flag for normalization
MAX_TOKENS = 64
BATCH_SIZE = 8

import librosa
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
    audio, sr = librosa.load(audio_path, sr=48000)

    # Convert stereo to mono if necessary
    if audio.ndim == 2:  # Stereo audio
        audio = audio.mean(axis=1)  # Average the two channels

    # Normalize audio to the range [-1, 1] if required
    if NORMALIZING_INPUT:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    return audio, sr

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
        if sample_rate != 48000:
            raise ValueError(f"Invalid sample rate: {sample_rate}. Expected 48000 Hz.")
        
        inputs = self.processor(audios=processed_audio, return_tensors="pt", sampling_rate=sample_rate)

        # Tokenize caption
        labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS)

        return {
            "inputs": inputs,
            "labels": labels["input_ids"].squeeze(0),
            "decoder_attention_mask": labels["attention_mask"].squeeze(0)
        }