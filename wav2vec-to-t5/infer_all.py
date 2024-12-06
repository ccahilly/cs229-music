import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os
from dataset_helpers import AudioCaptionDataset, preprocess_audio, MAX_TOKENS, BATCH_SIZE
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

    t5_model = T5ForConditionalGeneration.from_pretrained(local_path + "/t5").to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(local_path + "/t5")
    wav2vec_model = Wav2Vec2Model.from_pretrained(local_path + "/wav2vec").to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(local_path + "/wav2vec")
    reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, t5_model.config.d_model).to(DEVICE)
    reduce_layer.load_state_dict(torch.load(os.path.join(local_path + "/linear", "reduce_layer.pth")))

    delete_local_copy(local_path)

    return t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer

def infer(t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, audio_paths, batch_size=8):
    captions = []
    num_batches = len(audio_paths) // batch_size + (len(audio_paths) % batch_size != 0)

    for batch_idx in range(num_batches):
        # Get batch of audio paths
        batch_audio_paths = audio_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        # Preprocess audio files in the batch
        batch_inputs = []
        valid_paths = []
        for audio_path in batch_audio_paths:
            processed_audio, sample_rate = preprocess_audio(audio_path)
            if processed_audio is not None and sample_rate == 16000:
                inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
                batch_inputs.append(inputs["input_values"])
                valid_paths.append(audio_path)
            else:
                print(f"Skipping {audio_path} due to invalid preprocessing.")
        
        if not batch_inputs:  # Skip if no valid inputs
            continue

        # Combine inputs into a single tensor for batch processing
        batch_inputs = torch.cat(batch_inputs, dim=0).to(DEVICE)

        with torch.no_grad():
            # Pass batch inputs through the Wav2Vec model
            wav2vec_outputs = wav2vec_model(input_values=batch_inputs)
            audio_embeddings = wav2vec_outputs.last_hidden_state  # Shape: [batch_size, seq_len, feature_dim]

            # Reduce embeddings
            reduced_embeddings = reduce_layer(audio_embeddings)  # Shape: [batch_size, seq_len, t5_dim]

            # Generate captions
            generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings, max_new_tokens=MAX_TOKENS)
            batch_captions = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            
            # Append captions with corresponding audio paths
            captions.extend(batch_captions)

        print(f"Processed batch {batch_idx + 1}/{num_batches}")

    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=str, default="frozen", help="frozen or unfrozen")
    parser.add_argument("--epoch", type=int, default=15, help="Model checkpoint epoch to load")
    args = parser.parse_args()

    model_name = "fine_tuned_wav2vec_t5_"
    model_name += args.frozen

    batch_size = 8

    # Load test metadata
    test_metadata = pd.read_csv(test_data_path)
    output_file = "generated_captions_" + args.frozen + f"_e{args.epoch}" + ".pkl"

    for metadata in [test_metadata]:
        epoch = args.epoch
        print(f"Processing epoch {epoch}...")
        
        # Load model components
        t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer = load_model_checkpoint(model_name, epoch)
        
        # Perform inference
        generated_captions = infer(
            t5_model, t5_tokenizer, wav2vec_model, processor, reduce_layer, metadata["file_path"]
        )

        # Combine true captions and generated captions for saving
        combined_captions = [
            (true_caption, generated_caption)
            for true_caption, generated_caption in zip(metadata["caption"], generated_captions)
        ]
        # print(combined_captions)
        
        # Save combined captions to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(combined_captions, f)
        
        print(f"True and generated captions saved to {output_file}")

if __name__ == "__main__":
    main()