import json
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader, Dataset
from dpo.generate_pairs import SAMPLE_RATE
from scipy.io.wavfile import read
import numpy as np
from generate_human_feedback import feedback_file
from generate_pairs import reference_model_name, policy_model_name, logprobs_dir, REF_IDX, POL_IDX, iteration_number

output_dir = f"../models/musicgen-{iteration_number + 1}"

batch_size = 16
learning_rate = 5e-5
num_epochs = 3
BETA = 0.1

# Prepare Dataset
class HumanFeedbackDataset(Dataset):
    def __init__(self, feedback_data):
        self.data = feedback_data
        self.logprobs_dir = logprobs_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load the logprobs for the ref and pol models
        ref_logprob = np.load(f"{self.logprobs_dir}/{item['ytid']}-temp{item['temp']}-pair{item['pair_idx']}-{REF_IDX}.npy")
        pol_logprob = np.load(f"{self.logprobs_dir}/{item['ytid']}-temp{item['temp']}-pair{item['pair_idx']}-{POL_IDX}.npy")
        
        return {
            # "ytid": item["ytid"],
            # "temp": item["temp"],
            "preference": item["preference"],
            "ref_logprob": torch.tensor(ref_logprob, dtype=torch.float32),
            "pol_logprob": torch.tensor(pol_logprob, dtype=torch.float32),
        }

# Define DPO Loss
import torch.nn.functional as F

def dpo_loss(pol_logps, ref_logps, prefs, beta=BETA):
    """
    pol_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    prefs: human preferences, either POL_IDX or REF_IDX, shape (B,)
    beta: temperature controlling strength of KL penalty
    """
    # Compute the log ratio between policy and reference model logprobs directly
    pol_logratios = pol_logps[prefs == POL_IDX] - pol_logps[prefs != POL_IDX]
    ref_logratios = ref_logps[prefs == REF_IDX] - ref_logps[prefs != REF_IDX]
    
    # Compute the DPO loss as a sigmoid cross-entropy of the logratios, scaled by beta
    losses = -F.logsigmoid(beta * (pol_logratios - ref_logratios)).mean()
    
    # Calculate the reward term, which is the log difference between policy and reference model logprobs
    rewards = beta * (pol_logps - ref_logps).detach()
    
    return losses, rewards

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Human Feedback Data
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)

    dataset = HumanFeedbackDataset(feedback_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load MusicGen Model and Tokenizer
    policy_model = MusicgenForConditionalGeneration.from_pretrained(policy_model_name).to(device)
    policy_processor = AutoProcessor.from_pretrained(policy_model_name)

    # Training Loop
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

    policy_model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Compute DPO Loss
            loss, _ = dpo_loss(batch["pol_logprob"], batch["ref_logprob"], batch["preference"])

            # Backpropagation
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} completed. Loss: {loss.item()}")

    # Save Fine-Tuned Model
    policy_model.save_pretrained(output_dir)
    policy_processor.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    finetune()