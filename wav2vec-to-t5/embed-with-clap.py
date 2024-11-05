import os
import torch
import librosa
from transformers import AutoProcessor, AutoModel
import pandas as pd

SAMPLE_RATE = 48000 # 48 khz sampling

# Check if a GPU is available and use it; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
model = AutoModel.from_pretrained("laion/larger_clap_music").to(device)  # Send model to GPU

labels_df = pd.read_csv('../data/musiccaps/musiccaps-train-data.csv')
labels_dict = pd.Series(labels_df.caption.values, index=labels_df.ytid).to_dict()

# Directory containing the .wav files
# wav_dir = "../data/musiccaps/wav"
wav_dir = "../data/musiccaps/wav-small"

audio_samples = []
audio_filenames = [] 

for ytid in labels_df['ytid']:  # Loop through ytid to ensure we process all files in the CSV
    file_path = os.path.join(wav_dir, f"{ytid}.wav")  # Construct the file path
    if os.path.exists(file_path):  # Check if the audio file exists
        try:
            # Load the audio file and resample to 48 kHz
            audio, sr = librosa.load(file_path, sr=48000)  # Load with 48 kHz sampling rate
            if audio.size == 0:  # Check if the loaded audio is empty
                print(f"Warning: {ytid}.wav is empty after loading.")
                continue
            
            audio_samples.append(audio)
            audio_filenames.append(ytid)  # Store the ytid as the filename
            
        except Exception as e:
            print(f"Error loading {ytid}.wav: {e}")
    else:
        print(f"Warning: {ytid}.wav does not exist.")

# Prepare the audio_sample variable
audio_sample = {"audio": {"array": audio_samples}}

print("Done creating audio_sample dictionary")

# Process the audio samples; ensure they are on the correct device
inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt", sampling_rate = SAMPLE_RATE ).to(device)

# Get audio features
with torch.no_grad():  # Disable gradient calculation for inference
    audio_embed = model.get_audio_features(**inputs)

embedding_dict = {
    "filenames": audio_filenames,
    "embeddings": audio_embed.cpu().numpy(),  # Convert to numpy if necessary
    "labels": [labels_dict.get(ytid, None) for ytid in audio_filenames]  # Get corresponding labels from caption
}

if wav_dir == "../data/musiccaps/wav-small":
    out_file = 'audio_embeddings_with_labels_small.pt'
else:
    out_file = 'audio_embeddings_with_labels.pt'

torch.save(embedding_dict, out_file)

# To load later:
# loaded_data = torch.load('audio_embeddings_with_labels.pt')
# embeddings = loaded_data["embeddings"]
# filenames = loaded_data["filenames"]
# labels = loaded_data["labels"]

# Print the keys of the embedding dictionary
print("Keys in the embedding dictionary:")
print(embedding_dict.keys())

# Print the number of embeddings
print(f"Number of embeddings: {len(embedding_dict['embeddings'])}")

# Print the first 5 filenames, embeddings, and labels for verification
print("\nFirst 5 entries in the embedding dictionary:")
for i in range(min(5, len(embedding_dict['embeddings']))):
    print(f"Filename: {embedding_dict['filenames'][i]}")
    print(f"Label: {embedding_dict['labels'][i]}")
    print(f"Embedding shape: {embedding_dict['embeddings'][i].shape}")  # Shape of the embedding
    print("-" * 40)  # Separator for clarity