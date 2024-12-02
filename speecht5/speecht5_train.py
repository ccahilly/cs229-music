import evaluate
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

SAMPLE_RATE = 16000
BATCH_SIZE = 8
NUM_EPOCHS = 5
NORMALIZING_INPUT = True

class SpeechDataset(Dataset):
    def __init__(self, data, processor, audio_dir="../data/wav"):
        self.data = data
        self.processor = processor
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.data.iloc[idx]["file_path"]
        
        # Load the audio file using scipy (or librosa if you prefer)
        _, audio = wavfile.read(audio_path)
        
        # Check if the audio is stereo (2 channels) and convert it to mono if necessary
        if audio.ndim == 2:  # Stereo audio (2 channels)
            audio = audio.mean(axis=1)  # Convert stereo to mono by averaging the channels

        if NORMALIZING_INPUT:
            # Convert audio to float and normalize to [-1, 1]
            audio = audio.astype(np.float32) / np.iinfo(np.int16).max
            
        # Use the processor to convert audio into input values (with proper sample rate)
        audio_input = self.processor(audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

        # Get the text caption (this is used as the target label for the model)
        caption = self.data.iloc[idx]["caption"]

        # Return processed audio and corresponding caption (label)
        return {"input_values": audio_input["input_values"].squeeze(), "labels": caption}

def train():
    # Paths
    split_save_path = "../data/splits"
    checkpoint_dir = "../models/checkpoints"  # Directory to save checkpoints
    if NORMALIZING_INPUT:
        checkpoint_dir += "_norm"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load the preprocessed data splits
    train_data = pd.read_csv(os.path.join(split_save_path, "train.csv"))
    val_data = pd.read_csv(os.path.join(split_save_path, "val.csv"))

    # 2. Prepare the processor and model for Speech-to-Text
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
    model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

    # 3. Move the model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # 4. Prepare DataLoader
    train_dataset = SpeechDataset(train_data, processor)
    val_dataset = SpeechDataset(val_data, processor)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Use WER (Word Error Rate) metric
    wer = evaluate.load("wer")

    # Store loss values for plotting
    # Store metrics for plotting
    train_losses, val_losses = [], []
    train_wer_scores, val_wer_scores = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        total_train_wer = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            input_values = batch["input_values"].to(device)  # Move input values to GPU
            labels = batch["labels"]

            # Tokenize the labels (captions) into IDs
            labels = processor(text_target=labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

            # Forward pass
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate train WER
            with torch.no_grad():
                predictions = outputs.logits.argmax(dim=-1)
                predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
                decoded_references = processor.batch_decode(labels, skip_special_tokens=True)
                total_train_wer += wer.compute(predictions=predicted_texts, references=decoded_references)

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_wer = total_train_wer / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_wer_scores.append(avg_train_wer)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss}, Train WER: {avg_train_wer}")

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_wer = 0
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"]

            # Tokenize the labels (captions) for validation
            labels = processor(text_target=labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                outputs = model(input_values, labels=labels)
                total_val_loss += outputs.loss.item()

                # Calculate WER
                predictions = outputs.logits.argmax(dim=-1)
                predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
                decoded_references = processor.batch_decode(labels, skip_special_tokens=True)
                total_val_wer += wer.compute(predictions=predicted_texts, references=decoded_references)

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_wer = total_val_wer / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_wer_scores.append(avg_val_wer)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss}, Validation WER: {avg_val_wer}")

        # Plot Train vs Validation Loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, marker='o', label="Train Loss")
        plt.plot(range(1, epoch + 2), val_losses, marker='o', label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        if NORMALIZING_INPUT:
            plt.savefig("loss_plot" + str(NUM_EPOCHS) + "_norm.jpg")
        else:
            plt.savefig("loss_plot_e" + str(NUM_EPOCHS) + ".jpg")
        plt.close()

        # Plot Train vs Validation WER
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_wer_scores, marker='o', label="Train WER")
        plt.plot(range(1, epoch + 2), val_wer_scores, marker='o', label="Validation WER")
        plt.xlabel("Epoch")
        plt.ylabel("WER")
        plt.title("Train and Validation WER Over Epochs")
        plt.legend()
        plt.grid(True)
        if NORMALIZING_INPUT:
            plt.savefig("wer_plot_e" + str(NUM_EPOCHS) + "_norm.jpg")
        else:
            plt.savefig("wer_plot_e" + str(NUM_EPOCHS) + ".jpg")
        plt.close()

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}")
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)

    # 8. Save the final model and processor
    final_model_path = os.path.join(checkpoint_dir, f"final_model_epoch_{NUM_EPOCHS}")
    if NORMALIZING_INPUT:
        model.save_pretrained(final_model_path + "_norm")
        processor.save_pretrained(final_model_path + "_norm")
    else:
        model.save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)

if __name__ == "__main__":
    train()