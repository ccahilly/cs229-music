import os
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, T5Tokenizer, T5ForConditionalGeneration, AutoModel
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
# from prep_all_data import data_dir
import torchaudio.transforms as T

data_dir = "../data/splits"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZING_INPUT = True  # Flag for normalization
DEBUG = False
MAX_TOKENS = 64

print("Device:", DEVICE)

mert_model_name = "m-a-p/MERT-v1-95M"
t5_model_name = "t5-small"

# old_model_save_path = "../models/fine_tuned_mert_t5_e1"
old_model_save_path = None

# Save the fine-tuned model
model_save_path = "../models/fine_tuned_mert_t5_e1"
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(model_save_path + "/mert", exist_ok=True)
os.makedirs(model_save_path + "/conv1d", exist_ok=True)
os.makedirs(model_save_path + "/linear", exist_ok=True)
os.makedirs(model_save_path + "/t5", exist_ok=True)

if old_model_save_path is None:
    # Load pretrained models
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
    
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

    # Define the linear and aggregator layers
    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
    reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)
else: # Using previously fine tuned
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(old_model_save_path + "/mert")
    mert_model = AutoModel.from_pretrained(old_model_save_path + "/mert").to(DEVICE)
    
    t5_tokenizer = T5Tokenizer.from_pretrained(old_model_save_path + "/t5")
    t5_model = T5ForConditionalGeneration.from_pretrained(old_model_save_path + "/t5").to(DEVICE)

    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
    aggregator.load_state_dict(torch.load(os.path.join(old_model_save_path + "/aggregator", "aggregator.pth")))

    reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(old_model_save_path + "/linear", "reduce_layer.pth")))

# def load_dataset(data_folder):
#     data = []
#     captions = []
#     metadata = pd.read_csv("../data/musiccaps-train-data.csv")
#     npy_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
#     for f in npy_files:
#         base_path = f.split("/")[-1]
#         ytid = base_path[:-4]
#         # print(ytid)
#         data.append({
#             "audio_array": np.load(f),
#             "ytid": ytid,
#             # "sampling_rate": mert_processor.desired_sampling_rate
#         })
#         captions.append(metadata[metadata["ytid"] == ytid]["caption"].values[0])
#     print(f"Example caption: {captions[0]}")
#     return Dataset.from_dict({"audio": data, "labels": captions})

def preprocess_audio(audio_path):
    """
    Preprocess audio file to ensure it is mono and normalized.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        np.ndarray: Preprocessed audio data.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != mert_processor.sampling_rate:
        # print(f"resampling from {sample_rate} to {mert_processor.sampling_rate}")
        resampler = T.Resample(orig_freq=sample_rate, new_freq=mert_processor.sampling_rate)
        waveform = resampler(waveform)

    # Convert stereo to mono if necessary
    if waveform.ndim == 2:  # Stereo audio
        waveform = waveform.mean(axis=0)  # Average the two channels

    # Normalize audio to the range [-1, 1] if required
    # if NORMALIZING_INPUT:
    #     waveform = waveform.astype(np.float32) / np.iinfo(np.int16).max

    waveform = waveform.squeeze().numpy()
    # print(f"Waveform type: {type(waveform)}, shape: {waveform.shape}")

    # print(f"mert {mert_processor.sampling_rate}")
    return waveform, mert_processor.sampling_rate

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

# Load data
train_dataset = AudioCaptionDataset(data_dir + "/train.csv", mert_processor, t5_tokenizer)
val_dataset = AudioCaptionDataset(data_dir + "/val.csv", mert_processor, t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Training function
def train(model, train_loader, val_loader, epochs):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs = batch["inputs"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            if DEBUG:
                # val = inputs["input_values"]
                print(f"inputs shape: {inputs.shape}")
                print(f"attention_mask shape: {attention_mask.shape}")
                print(f"labels shape: {labels.shape}")
                print(f"decoder_attention_mask shape: {decoder_attention_mask.shape}")

            # Extract embeddings
            with torch.no_grad():
                mert_outputs = mert_model(inputs, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()

                if DEBUG:
                    print(f"all_layer_hidden_states shape: {all_layer_hidden_states.shape}")
                
            combined_dim = all_layer_hidden_states.view(BATCH_SIZE, 13, -1)  # [batch_size, layers, time_steps * features]

            if DEBUG:
                print(f"combined_dim shape: {combined_dim.shape}")

            # Apply Conv1d for learnable aggregation
            aggregated_embedding = aggregator(combined_dim)  # [batch_size, 1, time_steps * features]

            if DEBUG:
                print(f"aggregated_embedding shape: {aggregated_embedding.shape}")

            # Uncombine the last dimension back into time_steps and features
            aggregated_embedding = aggregated_embedding.view(BATCH_SIZE, 749, 768)  # [batch_size, time_steps, features]

            if DEBUG:
                print(f"aggregated_embedding shape: {aggregated_embedding.shape}")

            # Reduce Wav2Vec2 embeddings
            reduced_embeddings = reduce_layer(aggregated_embedding)

            if DEBUG:
                print(f"reduced_embeddings shape: {reduced_embeddings.shape}")

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
        evaluate(model, val_loader)

    # Save the T5 model
    t5_model.save_pretrained(model_save_path + "/t5")

    # Save the Wav2Vec2 model
    mert_model.save_pretrained(model_save_path + "/mert")

    # Save the linear layer used for dimension reduction
    torch.save(reduce_layer.state_dict(), os.path.join(model_save_path + "/linear", "reduce_layer.pth"))
    torch.save(aggregator.state_dict(), os.path.join(model_save_path + "/aggregator", "aggregator.pth"))

    # Save the processor and tokenizer
    mert_processor.save_pretrained(model_save_path + "/mert")
    t5_tokenizer.save_pretrained(model_save_path + "/t5")

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch["inputs"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            mert_outputs = mert_model(inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()
            combined_dim = all_layer_hidden_states.view(BATCH_SIZE, 13, -1)
            aggregated_embedding = aggregator(combined_dim)  # [batch_size, 1, time_steps * features]
            aggregated_embedding = aggregated_embedding.view(BATCH_SIZE, 749, 768)
            reduced_embeddings = reduce_layer(aggregated_embedding)

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

if __name__ == "__main__":
    train(t5_model, train_loader, val_loader, EPOCHS)