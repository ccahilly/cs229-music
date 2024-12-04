import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import os
from wav2vec_to_t5_train import preprocess_audio
import random

# Define your device (cuda if available, else cpu)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)

# Function to compute the cosine similarity between two vectors
def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    """
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

# Function to embed all audio files in a directory and compute similarities
def analyze_audio_similarity(audio_directory):
    """
    Analyze audio files in the given directory, calculate embeddings using Wav2Vec2, 
    and compute cosine similarities.
    """
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith(".wav")]
    audio_files = random.sample(audio_files, 10)
    embeddings = []
    audio_paths = []

    # Embed each audio file
    for audio_file in audio_files:
        audio_path = os.path.join(audio_directory, audio_file)
        audio, sr = preprocess_audio(audio_path)

        # Process audio with Wav2Vec2 processor
        inputs = processor(audio, return_tensors="pt", sampling_rate=sr, padding=True).to(DEVICE)

        with torch.no_grad():
            # Get embeddings from the Wav2Vec2 model
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling to get 1D vector
            
        embeddings.append(embedding)
        # audio_paths.append(audio_path)

    # Calculate cosine similarities
    # cosine_similarities = {}
    cosine_similarities = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = compute_cosine_similarity(embeddings[i], embeddings[j])
            # cosine_similarities[(audio_paths[i], audio_paths[j])] = sim
            cosine_similarities.append(sim)

    # Print out cosine similarities
    # for (audio1, audio2), similarity in cosine_similarities.items():
    for similarity in cosine_similarities:
        # print(f"Cosine similarity between {audio1} and {audio2}: {similarity:.4f}")
        print(f"Cosine similarity: {similarity:.4f}")

if __name__ == "__main__":
    # Path to the directory containing your audio files
    audio_directory = "../data/wav"
    
    # Run the analysis
    analyze_audio_similarity(audio_directory)