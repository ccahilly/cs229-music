# Run from caption_generation directory with:
# python -m scripts.train

import torch
from torch.utils.data import DataLoader
import argparse
import os
from models import ClapT5Model
# from ..models.mert_t5_model import MertT5Model
# from ..models.wav2vec2_t5_model import Wav2Vec2T5Model
from transformers import AutoProcessor, T5Tokenizer
from tqdm import tqdm
from utils import parse_args, save_checkpoint, load_checkpoint, upload_to_gcs

# Evaluation function
def evaluate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Don't compute gradients during evaluation
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs, labels=labels)
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(val_loader)
    model.train()  # Set the model back to training mode
    return avg_loss

if __name__ == "__main__":
    # Setup & hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_GCP = False
    train_data_path = "../data/splits/train.csv"
    val_data_path = "../data/splits/val.csv"

    print("Device:", DEVICE)

    args = parse_args()
    EMBED_MODEL = args.embed_model
    FROZEN = args.frozen
    EPOCHS = args.epochs
    LAST_EPOCH = args.last_epoch
    LEARNING_RATE = args.learning_rate
    print(f"Training configuration: Embed Model = {EMBED_MODEL}, Frozen = {FROZEN}, Epochs = {EPOCHS}, Last Epoch = {LAST_EPOCH}, Learning Rate = {LEARNING_RATE}")

    model_save_path = f"checkpoints/{EMBED_MODEL}_t5_"
    gcloud_path = f"checkpoints/{EMBED_MODEL}_t5_"
    if FROZEN:
        model_save_path += "frozen"
        gcloud_path += "frozen"
    else:
        model_save_path += "unfrozen"
        gcloud_path += "unfrozen"
    os.makedirs(model_save_path, exist_ok=True)

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    if EMBED_MODEL == "clap":
        audio_processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
        model = ClapT5Model(DEVICE, frozen=FROZEN)
        from dataset import ClapAudioCaptionDataset as AudioCaptionDataset
        BATCH_SIZE = 8
    
    elif EMBED_MODEL == "mert":
        embed_model_name = "m-a-p/MERT-v1-95M"
        # from dataset import MertAudioCaptionDataset as AudioCaptionDataset
    elif EMBED_MODEL == "wav2vec2":
        embed_model_name = "facebook/wav2vec2-base-960h"
        # from dataset import Wav2Vec2AudioCaptionDataset as AudioCaptionDataset
    else:
        raise ValueError("Invalid embedding model specified.")

    # Load dataset
    train_dataset = AudioCaptionDataset(train_data_path, audio_processor, t5_tokenizer)
    val_dataset = AudioCaptionDataset(val_data_path, audio_processor, t5_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Load checkpoint if available
    if LAST_EPOCH != 0:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, model_save_path + f"checkpoint{LAST_EPOCH}.pth")

    # Training loop
    for epoch in range(LAST_EPOCH + 1, LAST_EPOCH + EPOCHS + 1):
        model.train()  # Ensure the model is in training mode
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
    
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch}/{LAST_EPOCH + EPOCHS} Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}")

        # Save the model checkpoint
        checkpoint_name = f"/checkpoint{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, avg_val_loss, model_save_path + checkpoint_name)
        if USE_GCP:
            upload_to_gcs(model_save_path + checkpoint_name, gcloud_path + checkpoint_name, delete_locally=False)