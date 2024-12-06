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

def infer(t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator, audio_paths, batch_size=8):
    captions = []
    num_batches = len(audio_paths) // batch_size + (len(audio_paths) % batch_size != 0)
    
    for batch_idx in range(num_batches):
        batch_audio_paths = audio_paths[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        # Preprocess all audio files in the batch
        batch_inputs = []
        for audio_path in batch_audio_paths:
            processed_audio, sample_rate = preprocess_audio(audio_path, processor)
            input = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
            batch_inputs.append(input["input_values"].squeeze(1))
        
        # Combine all inputs in the batch
        batch_inputs = torch.cat(batch_inputs, dim=0).to(DEVICE)

        if DEBUG:
            print(f"Batch inputs shape: {batch_inputs.shape}")

        with torch.no_grad():
            # Process through the MERT model
            mert_outputs = mert_model(batch_inputs, output_hidden_states=True)
            all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).permute(1, 0, 2, 3)  # Adjust dims
            combined_dim = all_layer_hidden_states.view(len(batch_audio_paths), 13, -1)
            
            # Aggregate embeddings
            aggregated_embedding = aggregator(combined_dim)  # [batch_size, 1, time_steps * features]
            aggregated_embedding = aggregated_embedding.view(len(batch_audio_paths), 749, 768)

            # Reduce embeddings
            reduced_embeddings = reduce_layer(aggregated_embedding)

            # Generate captions
            generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings, max_new_tokens=MAX_TOKENS)
            batch_captions = [t5_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            
            # Collect captions for the batch
            captions.extend(batch_captions)  # Append only the captions
        
        print(f"Processed batch {batch_idx + 1}/{num_batches}")
    
    return captions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=str, default="frozen", help="frozen or unfrozen")
    parser.add_argument("--epoch", type=int, default=15, help="Model checkpoint epoch to load")
    args = parser.parse_args()

    model_name = "fine_tuned_mert_pro_t5_"
    model_name += args.frozen
    
    output_file = "generated_captions_" + args.frozen + f"_e{args.epoch}" + ".pkl"

    if args.frozen:
        batch_size = 8
    else:
        batch_size = 4

    # Load test metadata
    test_metadata = pd.read_csv(test_data_path)

    for metadata in [test_metadata]:
        epoch = args.epoch
        print(f"Processing epoch {epoch}...")
        
        # Load model components
        t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator = load_model_checkpoint(model_name, epoch)
        
        # Perform inference
        generated_captions = infer(
            t5_model, t5_tokenizer, mert_model, processor, reduce_layer, aggregator,
            metadata["file_path"], batch_size
        )

        # Combine true captions and generated captions for saving
        combined_captions = [
            (true_caption, generated_caption)
            for true_caption, generated_caption in zip(metadata["caption"], generated_captions)
        ]
        
        # Save combined captions to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(combined_captions, f)
        
        print(f"True and generated captions saved to {output_file}")

if __name__ == "__main__":
    main()