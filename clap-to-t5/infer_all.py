import torch
from transformers import AutoModel, AutoProcessor, T5Tokenizer, T5ForConditionalGeneration, ClapAudioModelWithProjection
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

# Load the fine-tuned model, clap2 model, linear layer, and processor
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"
test_data_path = "../data/splits/test.csv"

def load_model_checkpoint(model_name, epoch, frozen):
    """
    Load the models from a specific checkpoint.
    """
    checkpoint_path = f"models/{model_name}/e{epoch}"
    local_path = f"/tmp/{model_name}/e{epoch}"
    download_from_gcs(checkpoint_path, local_path)

    t5_model = T5ForConditionalGeneration.from_pretrained(local_path + "/t5").to(DEVICE)
    t5_tokenizer = T5Tokenizer.from_pretrained(local_path + "/t5")
    clap_model = ClapAudioModelWithProjection.from_pretrained(local_path + "/clap").to(DEVICE)
    processor = AutoProcessor.from_pretrained(local_path + "/clap")

    delete_local_copy(local_path)

    return t5_model, t5_tokenizer, clap_model, processor

def infer(t5_model, t5_tokenizer, clap_model, processor, audio_paths, frozen, batch_size=8):
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
            if processed_audio is not None and sample_rate == 48000:
                inputs = processor(audios=processed_audio, return_tensors="pt", sampling_rate=sample_rate)
                batch_inputs.append(inputs["input_features"])
                valid_paths.append(audio_path)
            else:
                print(f"Skipping {audio_path} due to invalid preprocessing.")
        
        if not batch_inputs:  # Skip if no valid inputs
            continue

        # Combine inputs into a single tensor for batch processing
        batch_inputs = torch.cat(batch_inputs, dim=0).to(DEVICE)

        with torch.no_grad():
            # Pass batch inputs through the clap model
            # if frozen:
            #     clap_outputs = clap_model.get_audio_features(batch_inputs)
            # else:
            outputs = clap_model(batch_inputs)
            clap_outputs = outputs.audio_embeds

            # Generate captions
            generated_ids = t5_model.generate(inputs_embeds=clap_outputs.unsqueeze(1), max_new_tokens=MAX_TOKENS)
            batch_captions = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            
            # Append captions with corresponding audio paths
            captions.extend(batch_captions)

        print(f"Processed batch {batch_idx + 1}/{num_batches}")

    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=str, default="frozen", help="frozen or unfrozen")
    parser.add_argument("--epoch", type=int, default=15, help="Model checkpoint epoch to load")
    parser.add_argument("--dataset", type=str, default="test", help="test, train, or val")
    args = parser.parse_args()

    model_name = "fine_tuned_clap_t5_"
    model_name += args.frozen

    # Load test metadata
    if args.dataset == "val":
        metadata = pd.read_csv(val_data_path)
        output_file = f"generated_captions_{args.frozen}_e{args.epoch}_{args.dataset}.pkl"
    elif args.dataset == "train":
        metadata = pd.read_csv(train_data_path)
        output_file = f"generated_captions_{args.frozen}_e{args.epoch}_{args.dataset}.pkl"
    else:
        metadata = pd.read_csv(test_data_path)
        output_file = f"generated_captions_{args.frozen}_e{args.epoch}.pkl"
    
    epoch = args.epoch
    print(f"Processing epoch {epoch}...")
    
    # Load model components
    t5_model, t5_tokenizer, clap_model, processor= load_model_checkpoint(model_name, epoch, args.frozen == "frozen")
    
    # Perform inference
    generated_captions = infer(
        t5_model, t5_tokenizer, clap_model, processor, metadata["file_path"], args.frozen == "frozen"
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