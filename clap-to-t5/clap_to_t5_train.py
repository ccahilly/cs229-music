import os
from transformers import AutoModel, AutoProcessor, T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from gcloud_helpers import upload_to_gcs
import argparse
from dataset_helpers import AudioCaptionDataset, BATCH_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Clap and T5 models for audio captioning.")
    
    # Adding arguments for epochs, last_epoch, and frozen status
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument('--last_epoch', type=int, default=0, help="The last epoch used for checkpointing.")
    parser.add_argument('--freeze_embed', type=bool, default=False, help="Set whether to freeze the embedding model (True/False).")

    return parser.parse_args()

# Paths
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False

print("Device:", DEVICE)

# Update variables based on command-line arguments
args = parse_args()
EPOCHS = args.epochs
last_epoch = args.last_epoch
FROZEN_EMBED = args.freeze_embed
print(f"Training configuration: Epochs = {EPOCHS}, Last Epoch = {last_epoch}, Freeze Embed = {FROZEN_EMBED}")

# Save the fine-tuned model
if FROZEN_EMBED:
    model_save_path = "../models/fine_tuned_clap_t5_frozen"
    gcloud_path = "models/fine_tuned_clap_t5_frozen"
else:
    model_save_path = "../models/fine_tuned_clap_t5_unfrozen"
    gcloud_path = "models/fine_tuned_clap_t5_unfrozen"
os.makedirs(model_save_path, exist_ok=True)

model_name = "laion/larger_clap_music"

if last_epoch == 0:
    # Load pretrained models
    processor = AutoProcessor.from_pretrained(model_name)
    clap_model = AutoModel.from_pretrained(model_name).to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

else: # Using previously fine tuned
    model_name = "../models/fine_tuned_clap_t5"
    if FROZEN_EMBED:
        model_name += "_frozen"
    else:
        model_name += "_unfrozen"
    
    model_name += f"/e{last_epoch}"

    t5_model = T5ForConditionalGeneration.from_pretrained(model_name + "/t5").to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name + "/t5")

    clap_model = AutoModel.from_pretrained(model_name + "/clap").to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_name + "/clap")

# Load data
train_dataset = AudioCaptionDataset(train_data_path, processor, t5_tokenizer)
val_dataset = AudioCaptionDataset(val_data_path, processor, t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Training function
def train(model, clap_model, train_loader, val_loader, epochs):
    for param in clap_model.parameters():
        param.requires_grad = not FROZEN_EMBED # true when frozen is false
    for param in model.parameters():
        param.requires_grad = True

    if not FROZEN_EMBED:
        clap_model.train()
    else:
        clap_model.eval()

    model.train()

    if FROZEN_EMBED:
        optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=LEARNING_RATE
        )
    else:
        optimizer = torch.optim.AdamW(
        list(clap_model.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
        )

    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs = batch["inputs"].to(DEVICE)
            inputs["input_features"] = inputs["input_features"].squeeze(1)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            if DEBUG:
                print("Input:", inputs["input_features"].shape)
                print("Labels shape:", labels.shape)
                print("Decoder attention mask shape:", decoder_attention_mask.shape)

            # Extract embeddings
            if FROZEN_EMBED:
                with torch.no_grad():
                    clap_outputs = clap_model.get_audio_features(**inputs)
            else:
                clap_outputs = clap_model.get_audio_features(**inputs)
            
            if DEBUG:
                print("Clap last hidden state shape:", clap_outputs.shape)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=clap_outputs.unsqueeze(1),
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
        avg_val_loss = evaluate(model, clap_model, val_loader)

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

        # Save the Clap model
        os.makedirs(checkpoint_path + "/clap", exist_ok=True)
        processor.save_pretrained(checkpoint_path + "/clap")
        clap_model.save_pretrained(checkpoint_path + "/clap")
        upload_to_gcs(checkpoint_path + "/clap", gcloud_checkpoint_path + "/clap")

# Evaluation function
def evaluate(model, clap_model, val_loader):
    if not FROZEN_EMBED:
        clap_model.eval()
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch["inputs"].to(DEVICE)
            inputs["input_features"] = inputs["input_features"].squeeze(1)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            clap_outputs = clap_model.get_audio_features(**inputs)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=clap_outputs.unsqueeze(1),
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss = {avg_val_loss}")
    
    model.train()
    if not FROZEN_EMBED:
        clap_model.train()
    
    return avg_val_loss

if __name__ == "__main__":
    # Now call your train function
    train(t5_model, clap_model, train_loader, val_loader, EPOCHS)