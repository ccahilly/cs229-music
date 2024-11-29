import evaluate
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Paths
split_save_path = "../data/splits"

# 1. Load the preprocessed data splits
train_data = pd.read_csv(os.path.join(split_save_path, "train.csv"))
val_data = pd.read_csv(os.path.join(split_save_path, "val.csv"))
test_data = pd.read_csv(os.path.join(split_save_path, "test.csv"))

# 2. Prepare the processor and model for Speech-to-Text
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

# 3. Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# 4. Prepare the dataset class
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
        sample_rate, audio = wavfile.read(audio_path)
        
        # Use the processor to convert audio into input values (with proper sample rate)
        audio_input = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")

        # Get the text caption (this is used as the target label for the model)
        caption = self.data.iloc[idx]["caption"]

        # Return processed audio and corresponding caption (label)
        return {"input_values": audio_input["input_values"].squeeze(), "labels": caption}

# 5. Prepare DataLoader
train_dataset = SpeechDataset(train_data, processor)
val_dataset = SpeechDataset(val_data, processor)
test_dataset = SpeechDataset(test_data, processor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 6. Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Use WER (Word Error Rate) metric
wer = evaluate.load("wer")

num_epochs = 3

# Store loss values for plotting
epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_values = batch["input_values"].to(device)  # Move input values to GPU
        labels = batch["labels"]

        # Tokenize the labels (captions) into IDs
        labels = processor(labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

        # Forward pass
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} - Loss: {avg_loss}")

    # Validation
    model.eval()
    total_wer = 0
    for batch in tqdm(val_dataloader, desc="Validation"):
        input_values = batch["input_values"].to(device)  # Move input values to GPU
        labels = batch["labels"]
        
        # Tokenize the labels (captions) for validation
        labels = processor(labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_values, labels=labels)
        
        # Calculate WER
        predictions = outputs.logits.argmax(dim=-1)
        predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        total_wer += wer.compute(predictions=predicted_texts, references=labels)

    print(f"Validation WER: {total_wer / len(val_dataloader)}")

# 7. Plot the loss graph
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label="Training Loss")
plt.title("Training Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("training_loss_graph.png")

# 8. Save the model and processor
model.save_pretrained("./speecht5-model")
processor.save_pretrained("./speecht5-model")
