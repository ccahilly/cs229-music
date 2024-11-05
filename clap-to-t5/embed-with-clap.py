import os
import torch
import librosa
from transformers import AutoProcessor, AutoModel
import pandas as pd
from google.cloud import storage

SAMPLE_RATE = 48000 # 48 khz sampling

# Check if a GPU is available and use it; otherwise, fall back to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available; using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available; using CPU.")

# Load processor and model
processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
model = AutoModel.from_pretrained("laion/larger_clap_music").to(device)  # Send model to GPU

# Directory containing the 49 khz .wav files
# Google Cloud Storage setup

bucket_name = "musiccaps-wav-16khz"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
gcs_wav_dir = "wav-small"  # Update if your audio files are in a subdirectory within the bucket

# Get labels
# Download the CSV file from GCS to a temporary file
temp_csv_file = "/tmp/temp_musiccaps.csv"
blob = bucket.blob("musiccaps-train-data.csv")
blob.download_to_filename(temp_csv_file)

# Read the CSV file using pandas
labels_df = pd.read_csv(temp_csv_file)
labels_dict = pd.Series(labels_df.caption.values, index=labels_df.ytid).to_dict()

# Remove the temporary CSV file
os.remove(temp_csv_file)

audio_samples = []
audio_filenames = [] 

for ytid in labels_df['ytid']:  # Loop through ytid to ensure we process all files in the CSV
    file_name = f"{ytid}.wav"
    blob = bucket.blob(os.path.join(gcs_wav_dir, file_name))
    
    if blob.exists():  # Check if the audio file exists
        try:
            # Download the audio file from GCS to a temporary file
            with open(file_name, "wb") as f:
                blob.download_to_file(f)

            # Load the audio file and resample to 48 kHz
            audio, sr = librosa.load(file_name, sr=48000)  # Load with 48 kHz sampling rate
            if audio.size == 0:  # Check if the loaded audio is empty
                print(f"Warning: {ytid}.wav is empty after loading.")
                continue
            
            audio_samples.append(audio)
            audio_filenames.append(ytid)  # Store the ytid as the filename

            os.remove(file_name)  # Remove the temporary file
            
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

# Save dict to temp file and upload to GCS

gcs_output_file = "audio_embeddings_with_labels.pt"  # Desired filename in GCS
blob = bucket.blob(gcs_output_file)
temp_file = "/tmp/temp_embeddings.pt" 
torch.save(embedding_dict, temp_file)

# Upload the temporary file to GCS
blob.upload_from_filename(temp_file)

# Remove the temporary file
os.remove(temp_file)
print(f"Embedding dictionary saved to GCS: gs://{bucket_name}/{gcs_output_file}")


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