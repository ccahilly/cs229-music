import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os
from wav2vec_to_t5_train import AudioCaptionDataset, preprocess_audio, MAX_TOKENS
import pandas as pd

model_save_path = "../models/fine_tuned_wav2vec_t5_e2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load the fine-tuned model, Wav2Vec2 model, linear layer, and processor
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"
test_data_path = "../data/splits/test.csv"

t5_model = T5ForConditionalGeneration.from_pretrained(model_save_path + "/t5").to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained(model_save_path + "/t5")

wav2vec_model = Wav2Vec2Model.from_pretrained(model_save_path + "/wav2vec").to(DEVICE)
processor = Wav2Vec2Processor.from_pretrained(model_save_path + "/wav2vec")

reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, t5_model.config.d_model).to(DEVICE)
reduce_layer.load_state_dict(torch.load(os.path.join(model_save_path + "/linear", "reduce_layer.pth")))

test_dataset = AudioCaptionDataset(test_data_path, processor, t5_tokenizer)

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

def infer(audio_paths):
    captions = []

    for audio_path in audio_paths:
        # Preprocess the audio
        processed_audio, sample_rate = preprocess_audio(audio_path)
        if processed_audio is None or sample_rate != 16000:
            print(f"Skipping {audio_path} due to invalid preprocessing.")
            continue

        # Process audio with the Wav2Vec2 processor
        inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)

        # Extract embeddings from Wav2Vec2
        with torch.no_grad():
            wav2vec_outputs = wav2vec_model(**inputs)
            audio_embeddings = wav2vec_outputs.last_hidden_state

            # Reduce the dimension to match T5 input size
            reduced_embeddings = reduce_layer(audio_embeddings)

            # Generate caption using T5
            generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings, max_new_tokens=MAX_TOKENS)
            generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            captions.append((audio_path, generated_caption))
    
    return captions

def gen_n_samples(metadata, n):
    # Verify file paths
    metadata = metadata[metadata["file_path"].apply(os.path.exists)]

    # Infer captions for 5 samples
    sample_audio_paths = metadata["file_path"].iloc[:n].tolist()
    results = infer(sample_audio_paths)

    for audio_path, caption in results:
        ytid = os.path.basename(audio_path).split(".")[0]
        print(f"\nCaption for {ytid}: {caption}")
        real_caption = metadata[metadata["ytid"] == ytid]["caption"].values[0]
        print(f"\nReal caption for {ytid}: {real_caption}")

if __name__ == "__main__":
    n = 3

    print(f"\nGenerating {n} samples for train\n")
    train_metadata = pd.read_csv(train_data_path)
    gen_n_samples(train_metadata, n)

    print(f"\nGenerating {n} samples for val\n")
    val_metadata = pd.read_csv(val_data_path)
    gen_n_samples(val_metadata, n)

    print(f"\nGenerating {n} samples for test\n")
    test_metadata = pd.read_csv(test_data_path)
    gen_n_samples(test_metadata, n)