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
from gcloud_helpers import upload_to_gcs
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Wav2Vec2 and T5 models for audio captioning.")
    
    # Adding arguments for epochs, last_epoch, and frozen status
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument('--last_epoch', type=int, default=0, help="The last epoch used for checkpointing.")
    parser.add_argument('--frozen', type=bool, default=False, help="Set whether to freeze the embedding model (True/False).")
    
    return parser.parse_args()

# Paths
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZING_INPUT = True  # Flag for normalization
DEBUG = False
MAX_TOKENS = 64

print("Device:", DEVICE)

# Update variables based on command-line arguments
args = parse_args()
EPOCHS = args.epochs
last_epoch = args.last_epoch
FROZEN = args.frozen
print(f"Training configuration: Epochs = {EPOCHS}, Last Epoch = {last_epoch}, Frozen = {FROZEN}")

# Save the fine-tuned model
if FROZEN:
    model_save_path = "../models/fine_tuned_wav2vec_t5_frozen"
    gcloud_path = "models/fine_tuned_wav2vec_t5_frozen"
else:
    model_save_path = "../models/fine_tuned_wav2vec_t5_unfrozen"
    gcloud_path = "models/fine_tuned_wav2vec_t5_unfrozen"
os.makedirs(model_save_path, exist_ok=True)

# model_name = "facebook/wav2vec2-large-960h"
model_name = "facebook/wav2vec2-base-960h"

if last_epoch == 0:
    # Load pretrained models
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
    reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, t5_model.config.d_model).to(DEVICE)

else: # Using previously fine tuned
    model_name = "../models/fine_tuned_wav2vec_t5"
    if FROZEN:
        model_name += "_frozen"
    else:
        model_name += "_unfrozen"
    
    model_name += f"/e{last_epoch}"

    t5_model = T5ForConditionalGeneration.from_pretrained(model_name + "/t5").to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name + "/t5")

    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name + "/wav2vec").to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(model_name + "/wav2vec")

    reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(model_name + "/linear", "reduce_layer.pth")))

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

# Load data
train_dataset = AudioCaptionDataset(train_data_path, processor, t5_tokenizer)
val_dataset = AudioCaptionDataset(val_data_path, processor, t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Training function
def train(model, wav2vec_model, train_loader, val_loader, epochs):
    for param in wav2vec_model.parameters():
        param.requires_grad = not FROZEN # true when frozen is false
    for param in reduce_layer.parameters():
        param.requires_grad = True
    for param in model.parameters():
        param.requires_grad = True
    
    if not FROZEN:
        wav2vec_model.train()
    else:
        wav2vec_model.eval()

    reduce_layer.train()
    model.train()

    if FROZEN:
        optimizer = torch.optim.AdamW(
        list(reduce_layer.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
        )
    else:
        optimizer = torch.optim.AdamW(
        list(wav2vec_model.parameters()) + list(reduce_layer.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
        )

    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            if DEBUG:
                print("Input values shape:", input_values.shape)
                print("Attention mask shape:", attention_mask.shape)
                print("Labels shape:", labels.shape)
                print("Decoder attention mask shape:", decoder_attention_mask.shape)

            # Extract embeddings
            if FROZEN:
                with torch.no_grad():
                    wav2vec_outputs = wav2vec_model(input_values, attention_mask=attention_mask)
            else:
                wav2vec_outputs = wav2vec_model(input_values, attention_mask=attention_mask)
            
            audio_embeddings = wav2vec_outputs.last_hidden_state
            if DEBUG:
                print("Wav2Vec2 last hidden state shape:", audio_embeddings.shape)

            # Reduce Wav2Vec2 embeddings
            reduced_embeddings = reduce_layer(audio_embeddings)

            if DEBUG:
                print("Reduced embeddings shape:", reduced_embeddings.shape)
                print("Expected T5 embedding size:", t5_model.config.d_model)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=reduced_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            if DEBUG:
                print("T5 output logits shape (if available):", outputs.logits.shape if hasattr(outputs, 'logits') else "Not available")
                print("T5 loss:", outputs.loss.item())

            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss}")

        # Evaluate
        avg_val_loss = evaluate(model, wav2vec_model, val_loader)

        checkpoint_path = model_save_path + f"/e{last_epoch + epoch + 1}"
        gcloud_checkpoint_path = gcloud_path + f"/e{last_epoch + epoch + 1}"
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the loss
        with open(checkpoint_path + "/loss.txt", "w") as f:
            f.write(f"Epoch {last_epoch + epoch + 1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}\n")
        upload_to_gcs(checkpoint_path + "/loss.txt", gcloud_checkpoint_path + "/loss.txt")
    
        # Save the T5 model
        os.makedirs(checkpoint_path + "/t5", exist_ok=True)
        t5_tokenizer.save_pretrained(checkpoint_path + "/t5")
        model.save_pretrained(checkpoint_path + "/t5")
        upload_to_gcs(checkpoint_path + "/t5", gcloud_checkpoint_path + "/t5")

        # Save the Wav2Vec2 model
        os.makedirs(checkpoint_path + "/wav2vec", exist_ok=True)
        processor.save_pretrained(checkpoint_path + "/wav2vec")
        wav2vec_model.save_pretrained(checkpoint_path + "/wav2vec")
        upload_to_gcs(checkpoint_path + "/wav2vec", gcloud_checkpoint_path + "/wav2vec")

        # Save the linear layer used for dimension reduction
        os.makedirs(checkpoint_path + "/linear", exist_ok=True)
        torch.save(reduce_layer.state_dict(), checkpoint_path + "/linear" + "/reduce_layer.pth")
        upload_to_gcs(checkpoint_path + "/linear", gcloud_checkpoint_path + "/linear")

# Evaluation function
def evaluate(model, wav2vec_model, val_loader):
    model.eval()
    reduce_layer.eval()
    if not FROZEN:
        wav2vec_model.eval()

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

            # Reduce Wav2Vec2 embeddings
            reduced_embeddings = reduce_layer(audio_embeddings)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=reduced_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss}")
    
    model.train()
    reduce_layer.train()
    if not FROZEN:
        wav2vec_model.train()
    
    return avg_val_loss

if __name__ == "__main__":
    # Now call your train function
    train(t5_model, wav2vec_model, train_loader, val_loader, EPOCHS)