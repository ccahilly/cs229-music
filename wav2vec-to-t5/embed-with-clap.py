import os
import torch
import librosa
from transformers import AutoProcessor, AutoModel

SAMPLE_RATE = 48000 # 48 khz sampling

# Check if a GPU is available and use it; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
model = AutoModel.from_pretrained("laion/larger_clap_music").to(device)  # Send model to GPU

# Directory containing the .wav files
wav_dir = "../data/musiccaps/wav"
# wav_dir = "../data/musiccaps/wav-small"

audio_samples = []
for filename in os.listdir(wav_dir):
    if filename.endswith('.wav'):
        # print("Read file: " + filename)
        file_path = os.path.join(wav_dir, filename)
        # Load the audio file (this returns the audio time series and the sampling rate)
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE )  # Want to resample to 48khz b/c that's what CLAP expects
        audio_samples.append(audio)

# Prepare the audio_sample variable
audio_sample = {"audio": {"array": audio_samples}}

print("Done creating audio_sample dictionary")

# Process the audio samples; ensure they are on the correct device
inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt", sampling_rate = SAMPLE_RATE ).to(device)

# Get audio features
with torch.no_grad():  # Disable gradient calculation for inference
    audio_embed = model.get_audio_features(**inputs)

# Save the embeddings as a PyTorch tensor
torch.save(audio_embed.cpu(), wav_dir + "/../" + 'audio_embeddings.pt')  # Move to CPU before saving

# Print the shape of the embeddings
print("Shape of audio_embed: ", audio_embed.shape)

# loaded_embeddings = torch.load('audio_embeddings.pt')
