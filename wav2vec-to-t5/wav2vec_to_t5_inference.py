import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os
from wav2vec_to_t5_train import AudioCaptionDataset, preprocess_audio
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load the fine-tuned model, Wav2Vec2 model, linear layer, and processor
model_save_path = "../models/fine_tuned_wav2vec_t5"
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
            print(audio_embeddings.shape)

            # Reduce the dimension to match T5 input size
            reduced_embeddings = reduce_layer(audio_embeddings)
            print(reduced_embeddings.shape)

            print("Expected T5 embedding size:", t5_model.config.d_model)
            print("Reduced embedding size:", reduced_embeddings.size(-1))
            print("Linear layer output shape:", reduced_embeddings.shape)
            print("Expected reduced dim:", t5_model.config.d_model) 

            # Generate caption using T5
            # attention_mask = torch.ones(reduced_embeddings.shape[:2], dtype=torch.long).to(DEVICE)
            # generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings)
            generated_ids = t5_model.generate(inputs_embeds=audio_embeddings)
            generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            captions.append((audio_path, generated_caption))
    
    return captions

if __name__ == "__main__":
    # Load test data
    test_metadata = pd.read_csv(test_data_path)

    # Verify file paths
    test_metadata = test_metadata[test_metadata["file_path"].apply(os.path.exists)]

    # Infer captions for 5 samples
    sample_audio_paths = test_metadata["file_path"].iloc[:5].tolist()
    results = infer(sample_audio_paths)

    print(results)

    for audio_path, caption in results:
        print(f"Caption for {os.path.basename(audio_path)}: {caption}")