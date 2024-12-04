import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load the fine-tuned model, Wav2Vec2 model, linear layer, and processor
model_save_path = "../models/fine_tuned_wav2vec_t5"
processor = Wav2Vec2Processor.from_pretrained(model_save_path)
t5_model = T5ForConditionalGeneration.from_pretrained(model_save_path).to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained(model_save_path)

# Load Wav2Vec2 model and linear layer
wav2vec_model = Wav2Vec2Model.from_pretrained(model_save_path).to(DEVICE)
reduce_layer = nn.Linear(wav2vec_model.config.hidden_size, 512).to(DEVICE)
reduce_layer.load_state_dict(torch.load(os.path.join(model_save_path, "reduce_layer.pth")))

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

def infer(audio_path):
    # Preprocess the audio
    processed_audio, sample_rate = preprocess_audio(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"Invalid sample rate: {sample_rate}. Expected 16000 Hz.")

    # Process audio with the Wav2Vec2 processor
    inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)

    # Extract embeddings from Wav2Vec2
    with torch.no_grad():
        wav2vec_outputs = wav2vec_model(**inputs)
        audio_embeddings = wav2vec_outputs.last_hidden_state

        # Optionally reduce the dimension to match T5 input size (if needed)
        reduced_embeddings = reduce_layer(audio_embeddings)

    # Use T5 for conditional generation
    generated_ids = t5_model.generate(inputs_embeds=reduced_embeddings)

    # Decode the generated ids to get the caption
    generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_caption

# Example inference
audio_path = "path_to_audio_file.wav"
generated_caption = infer(audio_path)
print(f"Generated caption: {generated_caption}")