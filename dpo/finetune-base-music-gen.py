import json
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader, Dataset
from dpo.generate_pairs import SAMPLE_RATE
from scipy.io.wavfile import read
import numpy as np
from generate_human_feedback import feedback_file
from generate_pairs import reference_model_name, policy_model_name, logits_dir, REF_IDX, POL_IDX

batch_size = 16
learning_rate = 5e-5
num_epochs = 3

# Prepare Dataset
class HumanFeedbackDataset(Dataset):
    def __init__(self, feedback_data):
        self.data = feedback_data
        self.logits_dir = logits_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load the logits for the ref and pol models
        ref_logits_path = f"{self.logits_dir}/{item['ytid']}-temp{item['temp']}-pair{item['pair_idx']}-{REF_IDX}.npy"
        pol_logits_path = f"{self.logits_dir}/{item['ytid']}-temp{item['temp']}-pair{item['pair_idx']}-{POL_IDX}.npy"
        
        ref_logits = np.load(ref_logits_path)
        pol_logits = np.load(pol_logits_path)
        
        return {
            "ytid": item["ytid"],
            "temp": item["temp"],
            "preference": item["preference"],
            "ref_logits": torch.tensor(ref_logits, dtype=torch.float32),
            "pol_logits": torch.tensor(pol_logits, dtype=torch.float32),
        }

def get_audio_array(audio_path):
    _, data = read(audio_path)

    # Normalize using the data type's range
    if np.issubdtype(data.dtype, np.integer):  # Check if data type is integer
        max_val = np.iinfo(data.dtype).max  # Max value for int type (e.g., 32767 for int16)
        min_val = np.iinfo(data.dtype).min  # Min value for int type (e.g., -32768 for int16)
        audio_array = data.astype("float32") / max(abs(max_val), abs(min_val))  # Normalize to [-1, 1]
    else:
        # If already float, assume it's normalized
        audio_array = data.astype("float32")

    return audio_array

# Define DPO Loss
def dpo_loss(preferred_logits, less_preferred_logits, beta=0.1):
    """Calculates the DPO loss."""
    difference = (preferred_logits - less_preferred_logits) / beta
    return -torch.log(torch.sigmoid(difference)).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Human Feedback Data
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)

    dataset = HumanFeedbackDataset(feedback_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load MusicGen Model and Tokenizer
    model = MusicgenForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    # Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Tokenize and encode inputs
            ytids = batch["ytid"]
            preferences = batch["preference"]

            # Generate audio embeddings for preferred and less-preferred pairs
            preferred_audio, less_preferred_audio = [], []
            for idx, pref in enumerate(preferences):
                audio_0_path = f"../data/audio/{ytids[idx]}-temp{batch['temp'][idx]}-pair{batch['pair_idx'][idx]}-0.wav"
                audio_1_path = f"../data/audio/{ytids[idx]}-temp{batch['temp'][idx]}-pair{batch['pair_idx'][idx]}-1.wav"

                # Ignore if pref == -1 (no preference)
                if pref == 0:
                    preferred_audio.append(get_audio_array(audio_0_path))
                    less_preferred_audio.append(get_audio_array(audio_1_path))
                elif pref == 1:
                    preferred_audio.append(get_audio_array(audio_1_path))
                    less_preferred_audio.append(get_audio_array(audio_0_path))

            # Convert audio to embeddings using MusicGen
            preferred_inputs = processor(raw_audio=preferred_audio, return_tensors="pt").to(device)
            less_preferred_inputs = processor(raw_audio=less_preferred_audio, return_tensors="pt").to(device)

            # Compute DPO Loss
            loss = dpo_loss(preferred_inputs, less_preferred_inputs)

            # Backpropagation
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

    # Save Fine-Tuned Model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    train()