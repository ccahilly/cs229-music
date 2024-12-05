import torch
from transformers import Wav2Vec2FeatureExtractor, T5Tokenizer, T5ForConditionalGeneration, AutoModel
from scipy.io import wavfile
import torch.nn as nn
import os
from dataset_helpers import AudioCaptionDataset, preprocess_audio, MAX_TOKENS
import pandas as pd
from gcloud_helpers import download_from_gcs, delete_local_copy
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

DEBUG = False

# Load the fine-tuned model, Wav2Vec2 model, linear layer, and processor
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"
test_data_path = "../data/splits/test.csv"

def load_model_checkpoint(model_name, epoch):
    """
    Load the models from a specific checkpoint.
    """
    checkpoint_path = f"models/{model_name}/e{epoch}"
    local_path = f"/tmp/{model_name}/e{epoch}"
    download_from_gcs(checkpoint_path, local_path)

    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(local_path + "/mert",trust_remote_code=True)
    mert_model = AutoModel.from_pretrained(local_path + "/mert",trust_remote_code=True).to(DEVICE)

    t5_tokenizer = T5Tokenizer.from_pretrained(local_path + "/t5")
    t5_model = T5ForConditionalGeneration.from_pretrained(local_path + "/t5").to(DEVICE)

    aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
    aggregator.load_state_dict(torch.load(os.path.join(local_path + "/aggregator", "aggregator.pth")))

    reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(local_path + "/linear", "reduce_layer.pth")))

    delete_local_copy(local_path)

    return t5_model, t5_tokenizer, mert_model, mert_processor, reduce_layer, aggregator

def infer(t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator, audio_paths, batch_size=1):
    captions = []
    for audio_path in audio_paths:
        processed_audio, sample_rate = preprocess_audio(audio_path, processor)
        input = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
        input["input_values"] = input["input_values"].squeeze(1)

        if DEBUG:
            print(f"processed_audio shape: {processed_audio.shape}")
            print(input["input_values"].shape)
        
        with torch.no_grad():
            mert_outputs = mert_model(input["input_values"], output_hidden_states=True)
            all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()
            if DEBUG:
                print(f"all_layer_hidden_states shape: {all_layer_hidden_states.shape}")
            combined_dim = all_layer_hidden_states.view(batch_size, 13, -1)
            if DEBUG:
                print(f"combined_dim shape: {combined_dim.shape}")
            aggregated_embedding = aggregator(combined_dim)  # [batch_size, 1, time_steps * features]
            if DEBUG:
                print(f"aggregated_embedding shape: {aggregated_embedding.shape}")
            aggregated_embedding = aggregated_embedding.view(batch_size, 749, 768)
            if DEBUG:
                print(f"aggregated_embedding shape: {aggregated_embedding.shape}")
            reduced_embeddings = reduce_layer(aggregated_embedding)
            if DEBUG:
                print(f"reduced_embeddings shape: {reduced_embeddings.shape}")

            generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings, max_new_tokens=MAX_TOKENS)
            generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            captions.append((audio_path, generated_caption))
    
    return captions

# Evaluation function
def evaluate(model, mert_model, aggregator, reduce_layer, val_loader, batch_size):
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
            combined_dim = all_layer_hidden_states.view(batch_size, 13, -1)
            aggregated_embedding = aggregator(combined_dim)  # [batch_size, 1, time_steps * features]
            aggregated_embedding = aggregated_embedding.view(batch_size, 749, 768)
            reduced_embeddings = reduce_layer(aggregated_embedding)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=reduced_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)

    return avg_val_loss

# Sample random examples once and reuse them across all epochs
def sample_examples(metadata, n):
    # Ensure file paths exist before sampling
    metadata = metadata[metadata["file_path"].apply(os.path.exists)]
    return metadata.sample(n=n).to_dict('records')  # Convert to list of dicts for ease of access

def load_or_sample_examples(metadata, n, file_name):
    # Check if the file exists
    if os.path.exists(file_name):
        print(f"Loading samples from {file_name}...")
        with open(file_name, 'rb') as f:
            samples = pickle.load(f)
    else:
        print(f"Sampling {n} examples and saving to {file_name}...")
        samples = sample_examples(metadata, n)
        with open(file_name, 'wb') as f:
            pickle.dump(samples, f)
    return samples

# Generate captions for sampled examples
def generate_captions(t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator, samples, batch_size):
    captions = []
    for sample in samples:
        audio_path = sample["file_path"]
        real_caption = sample["caption"]
        try:
            inferred_caption = infer(t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator, [audio_path])[0][1]
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            inferred_caption = "Error in inference"
        captions.append((audio_path, real_caption, inferred_caption))
    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=str, default="frozen", help="frozen or unfrozen")
    parser.add_argument("--n", type=int, default=20, help="Number of random examples to sample.")
    args = parser.parse_args()

    model_name = "fine_tuned_mert_pro_t5_"
    model_name += args.frozen
    n = args.n

    if args.frozen == "frozen":
        batch_size = 8
    else:
        batch_size = 4

    train_metadata = pd.read_csv(train_data_path)
    val_metadata = pd.read_csv(val_data_path)
    test_metadata = pd.read_csv(test_data_path)

    with open("inference_data_" + args.frozen + ".txt", "w") as file:
        # Pre-sample examples
        train_samples = load_or_sample_examples(train_metadata, n, f"../data/samples-{n}-train.pkl")
        val_samples = load_or_sample_examples(val_metadata, n, f"../data/samples-{n}-val.pkl")
        test_samples = load_or_sample_examples(test_metadata, n, f"../data/samples-{n}-test.pkl")

        for epoch in range(1, 16):  # Iterate through all epochs
        # for epoch in range(1, 2):  # Iterate through all epochs
            print(f"Processing epoch {epoch}...")
            t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator = load_model_checkpoint(model_name, epoch)
            test_dataset = AudioCaptionDataset(test_data_path, processor, t5_tokenizer)

            # Evaluate loss
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            loss = evaluate(t5_model, mert_model, aggregator, reduce_layer, test_loader, batch_size)
            file.write(f"Epoch {epoch}: Test Loss = {loss}\n")

            # Process samples
            for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
                file.write(f"Epoch {epoch}: {split_name} Examples:\n")
                captions = generate_captions(t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator, samples, batch_size)
                for audio_path, real_caption, inferred_caption in captions:
                    file.write(f"{audio_path}\n\nReal: {real_caption}\n\nGenerated: {inferred_caption}\n\n")

if __name__ == "__main__":
    main()