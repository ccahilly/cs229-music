# Run from caption_generation directory with:
# python -m scripts.train

import torch
from torch.utils.data import DataLoader
import argparse
import os
from models import ClapT5Model
# from ..models.mert_t5_model import MertT5Model
# from ..models.wav2vec2_t5_model import Wav2Vec2T5Model
from transformers import T5ForConditionalGeneration, ClapAudioModelWithProjection, AutoProcessor, T5Tokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning caption generation model.")
    
    # Adding arguments for epochs, last_epoch, and frozen status
    parser.add_argument('--embed_model', type=str, default="clap", help="clap, mert or wav2vec2.")
    parser.add_argument('--freeze', type=bool, default=False, help="Set whether to freeze the embedding model (True/False).")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument('--last_epoch', type=int, default=0, help="The last epoch used for checkpointing.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    
    return parser.parse_args()

# Setup & hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_data_path = "../data/splits/train.csv"

print("Device:", DEVICE)

args = parse_args()
EMBED_MODEL = args.embed_model
FROZEN = args.freeze
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
    if LAST_EPOCH == 0:
        model = ClapT5Model(DEVICE, frozen=FROZEN)
    elif FROZEN:
        model_name = f"checkpoints/{EMBED_MODEL}_t5_frozen/e{LAST_EPOCH}"
        model = ClapT5Model(DEVICE, 
                        t5_model=T5ForConditionalGeneration.from_pretrained(model_name + "/t5").to(DEVICE),
                        frozen=FROZEN)
    else:
        model_name = f"checkpoints/{EMBED_MODEL}_t5_unfrozen/e{LAST_EPOCH}"
        model = ClapT5Model(DEVICE, 
                        clap_model=ClapAudioModelWithProjection.from_pretrained(model_name + "/clap").to(DEVICE),
                        t5_model=T5ForConditionalGeneration.from_pretrained(model_name + "/t5").to(DEVICE),
                        frozen=FROZEN)
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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Example training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")