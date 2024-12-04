import os
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import torch.nn as nn

# Paths
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZING_INPUT = True  # Flag for normalization

print("Device:", DEVICE)

# model_name = "facebook/wav2vec2-large-960h"
model_name = "facebook/wav2vec2-base-960h"

# Load pretrained models
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

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
        
        inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt")

        # Tokenize caption
        labels = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

        # Check if attention_mask is present
        input_values = inputs["input_values"].squeeze(0)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_values))  # Default to ones if missing

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels["input_ids"].squeeze(0),
            "decoder_attention_mask": labels["attention_mask"].squeeze(0)
        }

# Load data
train_dataset = AudioCaptionDataset(train_data_path, processor, t5_tokenizer)
val_dataset = AudioCaptionDataset(val_data_path, processor, t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Training function
def train(model, wav2vec_model, train_loader, val_loader, optimizer, epochs):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            with torch.no_grad():
                wav2vec_outputs = wav2vec_model(input_values, attention_mask=attention_mask)
                audio_embeddings = wav2vec_outputs.last_hidden_state

                # Add a linear layer to match dimensions
                embedding_dim = wav2vec_model.config.hidden_size  # Wav2Vec2 embedding size
                reduced_dim = 512  # T5 input size

                reduce_layer = nn.Linear(embedding_dim, reduced_dim).to(DEVICE)

                # Reduce Wav2Vec2 embeddings
                reduced_embeddings = reduce_layer(audio_embeddings)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=reduced_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss}")

        # Evaluate
        evaluate(model, wav2vec_model, val_loader)

# Evaluation function
def evaluate(model, wav2vec_model, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            wav2vec_outputs = wav2vec_model(input_values, attention_mask=attention_mask)
            audio_embeddings = wav2vec_outputs.last_hidden_state

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=audio_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss}")
    model.train()

# Initialize optimizer
optimizer = torch.optim.AdamW(t5_model.parameters(), lr=LEARNING_RATE)

# Train the model
train(t5_model, wav2vec_model, train_loader, val_loader, optimizer, EPOCHS)