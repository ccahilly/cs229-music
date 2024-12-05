import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os
from dataset_helpers import AudioCaptionDataset, preprocess_audio, MAX_TOKENS, BATCH_SIZE
import pandas as pd
from gcloud_helpers import download_from_gcs, delete_local_copy, upload_to_gcs
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

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

    t5_model = T5ForConditionalGeneration.from_pretrained(local_path + "/t5").to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(local_path + "/t5")
    wav2vec_model = Wav2Vec2Model.from_pretrained(local_path + "/wav2vec").to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(local_path + "/wav2vec")
    reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(local_path + "/linear", "reduce_layer.pth")))

    delete_local_copy(local_path)

    return t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer

def preprocess_audio(audio_path):
    """
    Preprocess audio file to ensure it is mono and normalized.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        np.ndarray: Preprocessed audio data.
    """
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # Convert stereo to mono if necessary
    if audio.ndim == 2:  # Stereo audio
        audio = audio.mean(axis=1)  # Average the two channels

    # Normalize audio to the range [-1, 1] if required
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    return audio, sample_rate

def infer(t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, audio_paths):
    captions = []
    for audio_path in audio_paths:
        processed_audio, sample_rate = preprocess_audio(audio_path)
        if processed_audio is None or sample_rate != 16000:
            print(f"Skipping {audio_path} due to invalid preprocessing.")
            continue

        inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            wav2vec_outputs = wav2vec_model(**inputs)
            audio_embeddings = wav2vec_outputs.last_hidden_state
            reduced_embeddings = reduce_layer(audio_embeddings)
            generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings, max_new_tokens=MAX_TOKENS)
            generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            captions.append((audio_path, generated_caption))
    
    return captions

# Evaluation function
def evaluate(model, wav2vec_model, reduce_layer, test_loader):
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_values = batch["input_values"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            # Extract embeddings
            wav2vec_outputs = wav2vec_model(input_values, attention_mask=attention_mask)
            audio_embeddings = wav2vec_outputs.last_hidden_state

            # Reduce Wav2Vec2 embeddings
            reduced_embeddings = reduce_layer(audio_embeddings)

            # Feed embeddings to T5
            outputs = model(
                inputs_embeds=reduced_embeddings,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            test_loss += outputs.loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss = {avg_test_loss}")
    
    return avg_test_loss

# Sample random examples once and reuse them across all epochs
def sample_examples(metadata, n):
    # Ensure file paths exist before sampling
    metadata = metadata[metadata["file_path"].apply(os.path.exists)]
    return metadata.sample(n=n).to_dict('records')  # Convert to list of dicts for ease of access

# Generate captions for sampled examples
def generate_captions(t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, samples):
    captions = []
    for sample in samples:
        audio_path = sample["file_path"]
        real_caption = sample["caption"]
        try:
            inferred_caption = infer(t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, [audio_path])[0][1]
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

    model_name = "fine_tuned_wav2vec_t5_"
    model_name += args.frozen
    n = args.n

    train_metadata = pd.read_csv(train_data_path)
    val_metadata = pd.read_csv(val_data_path)
    test_metadata = pd.read_csv(test_data_path)

    with open("inference_data_" + args.frozen + ".txt", "w") as file:
        # Pre-sample examples
        train_samples = sample_examples(train_metadata, n)
        val_samples = sample_examples(val_metadata, n)
        test_samples = sample_examples(test_metadata, n)

        for epoch in range(1, 16):  # Iterate through all epochs
            print(f"Processing epoch {epoch}...")
            t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer = load_model_checkpoint(model_name, epoch)
            test_dataset = AudioCaptionDataset(test_data_path, processor, t5_tokenizer)

            # Evaluate loss
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
            loss = evaluate(t5_model, wav2vec_model, reduce_layer, test_loader)
            file.write(f"Epoch {epoch}: Test Loss = {loss}\n")

            # Process samples
            for split_name, samples in [("Train", train_samples), ("Val", val_samples), ("Test", test_samples)]:
                file.write(f"Epoch {epoch}: {split_name} Examples:\n")
                captions = generate_captions(t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, samples)
                for audio_path, real_caption, inferred_caption in captions:
                    file.write(f"{audio_path}\n\nReal: {real_caption}\n\nGenerated: {inferred_caption}\n\n")

if __name__ == "__main__":
    main()