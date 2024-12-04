import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model, T5Tokenizer, T5ForConditionalGeneration
from scipy.io import wavfile
import numpy as np

# Hyperparameters
BATCH_SIZE = 1  # Inference typically done one by one
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZING_INPUT = True  # Flag for normalization

# Load pretrained models
model_name = "facebook/wav2vec2-base-960h"

processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

# Preprocess audio (same as you did in training)
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
    if NORMALIZING_INPUT:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

    return audio, sample_rate

# Test data (replace this with actual test data)
test_data_path = "../data/splits/test.csv"  # Path to your test data CSV

# Load test data
test_data = pd.read_csv(test_data_path)

# Iterate over test data
for idx in range(5):  # For the first 5 samples
    row = test_data.iloc[idx]
    audio_path = row["file_path"]
    real_caption = row["caption"]

    # Load and preprocess the audio
    processed_audio, sample_rate = preprocess_audio(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"Invalid sample rate: {sample_rate}. Expected 16000 Hz.")
    
    # Use Wav2Vec2 to extract features
    inputs = processor(processed_audio, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        wav2vec_outputs = wav2vec_model(input_values=inputs["input_values"].to(DEVICE))
        audio_embeddings = wav2vec_outputs.last_hidden_state

    # Use T5 to generate captions
    input_ids = t5_tokenizer("generate a caption: ", return_tensors="pt").input_ids.to(DEVICE)  # Add some prompt
    generated_ids = t5_model.generate(inputs_embeds=audio_embeddings, decoder_input_ids=input_ids)

    # Decode the generated text
    generated_caption = t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Print the generated vs real caption
    print(f"Sample {idx + 1}:")
    print(f"Real Caption: {real_caption}")
    print(f"Generated Caption: {generated_caption}")
    print("-" * 50)