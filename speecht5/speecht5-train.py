import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SpeechT5Processor, SpeechT5ForConditionalGeneration
from datasets import load_metric
import librosa

# Paths
train_split = "../data/splits/train.csv"
val_split = "../data/splits/val.csv"
output_dir = "../model_weights"

# Load splits
train_data = pd.read_csv(train_split)
val_data = pd.read_csv(val_split)

# Define Dataset
class AudioCaptionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path, caption = row["file_path"], row["caption"]

        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000)  # Ensures 16kHz
        input_features = self.processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

        # Tokenize caption
        labels = self.processor.tokenizer(caption, return_tensors="pt", padding=True).input_ids

        return {
            "input_features": input_features["input_features"].squeeze(0),
            "labels": labels.squeeze(0),
        }

# Initialize processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForConditionalGeneration.from_pretrained("microsoft/speecht5_tts")

# Prepare DataLoader
train_dataset = AudioCaptionDataset(train_data, processor)
val_dataset = AudioCaptionDataset(val_data, processor)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # Move data to GPU if available
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

    # Validation step (optional)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_features=input_features, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")

# Save Model
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print("Model and processor saved!")