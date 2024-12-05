import os
from transformers import Wav2Vec2FeatureExtractor, T5Tokenizer, T5ForConditionalGeneration, AutoModel
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from gcloud_helpers import upload_to_gcs
import argparse
from dataset_helpers import AudioCaptionDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Wav2Vec2 and T5 models for audio captioning.")
    
    # Adding arguments for epochs, last_epoch, and frozen status
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument('--last_epoch', type=int, default=0, help="The last epoch used for checkpointing.")
    parser.add_argument('--frozen', type=bool, default=False, help="Set whether to freeze the embedding model (True/False).")
    
    return parser.parse_args()

data_dir = "../data/splits"

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = False

print("Device:", DEVICE)

# Update variables based on command-line arguments
args = parse_args()
EPOCHS = args.epochs
last_epoch = args.last_epoch
FROZEN = args.frozen
print(f"Training configuration: Epochs = {EPOCHS}, Last Epoch = {last_epoch}, Frozen = {FROZEN}")

if FROZEN:
    BATCH_SIZE = 8
else:
    BATCH_SIZE = 4

# Save the fine-tuned model
if FROZEN:
    model_save_path = "../models/fine_tuned_mert_pro_t5_frozen"
    gcloud_path = "models/fine_tuned_mert_pro_t5_frozen"
else:
    model_save_path = "../models/fine_tuned_mert_pro_t5_unfrozen"
    gcloud_path = "models/fine_tuned_mert_pro_t5_unfrozen"
os.makedirs(model_save_path, exist_ok=True)

mert_model_name = "m-a-p/MERT-v1-95M"
t5_model_name = "t5-small"

if last_epoch == 0:
    # Load pretrained models
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
    
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

    # Define the linear and aggregator layers
    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
    reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)

else: # Using previously fine tuned
    old_model_save_path = "../models/fine_tuned_mert_pro_t5"
    if FROZEN:
        old_model_save_path += "_frozen"
    else:
        old_model_save_path += "_unfrozen"
    
    old_model_save_path += f"/e{last_epoch}"

    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(old_model_save_path + "/mert")
    mert_model = AutoModel.from_pretrained(old_model_save_path + "/mert", trust_remote_code=True).to(DEVICE)
    
    t5_tokenizer = T5Tokenizer.from_pretrained(old_model_save_path + "/t5")
    t5_model = T5ForConditionalGeneration.from_pretrained(old_model_save_path + "/t5").to(DEVICE)

    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
    aggregator.load_state_dict(torch.load(os.path.join(old_model_save_path + "/aggregator", "aggregator.pth")))

    reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(old_model_save_path + "/linear", "reduce_layer.pth")))

# Load data
train_dataset = AudioCaptionDataset(data_dir + "/train.csv", mert_processor, t5_tokenizer)
val_dataset = AudioCaptionDataset(data_dir + "/val.csv", mert_processor, t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Training function
def train(model, train_loader, val_loader, epochs):
    for param in mert_model.parameters():
        param.requires_grad = not FROZEN # true when frozen is false
    for param in aggregator.parameters():
        param.requires_grad = True
    for param in reduce_layer.parameters():
        param.requires_grad = True
    for param in model.parameters():
        param.requires_grad = True
    
    if not FROZEN:
        mert_model.train()
    else:
        mert_model.eval()

    aggregator.train()
    reduce_layer.train()
    model.train()

    if FROZEN:
        optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(reduce_layer.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
        )
    else:
        optimizer = torch.optim.AdamW(
        list(mert_model.parameters()) + list(aggregator.parameters()) + list(reduce_layer.parameters()) + list(model.parameters()),
        lr=LEARNING_RATE
        )

    for epoch in range(epochs):
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs = batch["inputs"].to(DEVICE)
            inputs["input_values"] = inputs["input_values"].squeeze(1)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            if DEBUG:
                print(inputs["input_values"].shape)
                print(f"labels shape: {labels.shape}")
                print(f"decoder_attention_mask shape: {decoder_attention_mask.shape}")

            # Extract embeddings
            if FROZEN:
                with torch.no_grad():
                    mert_outputs = mert_model(inputs["input_values"], output_hidden_states=True)
            else:
                mert_outputs = mert_model(inputs["input_values"], output_hidden_states=True)
            
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
        avg_val_loss = evaluate(model, val_loader)

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

        # Save the MERT model
        os.makedirs(checkpoint_path + "/mert", exist_ok=True)
        mert_processor.save_pretrained(checkpoint_path + "/mert")
        mert_model.save_pretrained(checkpoint_path + "/mert")
        upload_to_gcs(checkpoint_path + "/mert", gcloud_checkpoint_path + "/mert")

        # Save the linear layer
        os.makedirs(checkpoint_path + "/linear", exist_ok=True)
        torch.save(reduce_layer.state_dict(), checkpoint_path + "/linear" + "/reduce_layer.pth")
        upload_to_gcs(checkpoint_path + "/linear", gcloud_checkpoint_path + "/linear")

        # Save the aggregator layer
        os.makedirs(checkpoint_path + "/aggregator", exist_ok=True)
        torch.save(aggregator.state_dict(), os.path.join(checkpoint_path + "/aggregator", "aggregator.pth"))
        upload_to_gcs(checkpoint_path + "/aggregator", gcloud_checkpoint_path + "/aggregator")

# Evaluation function
def evaluate(model, val_loader):
    model.eval()
    reduce_layer.eval()
    aggregator.eval()
    if not FROZEN:
        mert_model.eval()

    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = batch["inputs"].to(DEVICE)
            inputs["input_values"] = inputs["input_values"].squeeze(1)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            mert_outputs = mert_model(inputs["input_values"], output_hidden_states=True)
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
    reduce_layer.train()
    aggregator.train()
    if not FROZEN:
        mert_model.train()

    return avg_val_loss

if __name__ == "__main__":
    # Now call your train function
    train(t5_model, train_loader, val_loader, EPOCHS)