import os
import torch
import librosa
from transformers import AutoProcessor, AutoModel
import pandas as pd
from google.cloud import storage

CLOUD_DATA = False # Way lower latency when false.
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

if CLOUD_DATA:
    bucket_name = "musiccaps-wav-16khz"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_wav_dir = "wav-48"  # Update if your audio files are in a subdirectory within the bucket

    # Get labels
    # Download the CSV file from GCS to a temporary file
    temp_csv_file = "/tmp/temp_musiccaps.csv"
    blob = bucket.blob("musiccaps-train-data.csv")
    blob.download_to_filename(temp_csv_file)
else:
    temp_csv_file = "../data/musiccaps-train-data.csv"

# Read the CSV file using pandas
labels_df = pd.read_csv(temp_csv_file)
labels_dict = pd.Series(labels_df.caption.values, index=labels_df.ytid).to_dict()

# Remove the temporary CSV file
if CLOUD_DATA:
    os.remove(temp_csv_file)

audio_samples = []
audio_filenames = [] 

for ytid in labels_df['ytid']:  # Loop through ytid to ensure we process all files in the CSV
    file_name = f"{ytid}.wav"

    if CLOUD_DATA:
        blob = bucket.blob(os.path.join(gcs_wav_dir, file_name))
        exists = blob.exists()
    else:
        file_name = "../data/wav-48/" + file_name 
        exists = os.path.exists(file_name)

    if exists:  # Check if the audio file exists
        try:
            # Download the audio file from GCS to a temporary file
            if CLOUD_DATA:
                with open(file_name, "wb") as f:
                    blob.download_to_file(f)

            # Load the audio file and resample to 48 kHz    
            audio, sr = librosa.load(file_name, sr=48000)  # Load with 48 kHz sampling rate
            if audio.size == 0:  # Check if the loaded audio is empty
                print(f"Warning: {ytid}.wav is empty after loading.")
                continue
            
            audio_samples.append(audio)
            audio_filenames.append(ytid)  # Store the ytid as the filename

            if CLOUD_DATA:
                os.remove(file_name)  # Remove the temporary file
            
        except Exception as e:
            print(f"Error loading {ytid}.wav: {e}")
    # else:
    #     print(f"Warning: " + file_name + " does not exist.")

print(f"Number of audio samples: {len(audio_samples)}")

# Prepare the audio_sample variable
audio_sample = {"audio": {"array": audio_samples}}

print("Done creating audio_sample dictionary")
print(f"Audio sample shape: {len(audio_sample['audio']['array'])}")

batch_size = 250
num_batches = len(audio_sample["audio"]["array"]) // batch_size
batches = [audio_sample["audio"]["array"][i:i + batch_size] for i in range(0, len(audio_sample["audio"]["array"]), batch_size)]

all_inputs = []
all_audio_embeddings = []

for i, batch in enumerate(batches):
    # Process each batch: pass it to the processor
    inputs = processor(audios=batch, return_tensors="pt", sampling_rate=SAMPLE_RATE).to(device)
    
    print(f"Processing batch {i+1}/{len(batches)}")

    # Get audio features (embeddings)
    with torch.no_grad():  # Disable gradient calculation for inference
        audio_embed = model.get_audio_features(**inputs)

    # Store the results from this batch
    # all_inputs.append(inputs)
    all_audio_embeddings.append(audio_embed.cpu().numpy())

# Combine all embeddings into one list
combined_embeddings = []
for embeddings in all_audio_embeddings:
    combined_embeddings.extend(embeddings)

# Now you can save the combined embeddings or process them further
embedding_dict = {
    "filenames": audio_filenames,
    "embeddings": combined_embeddings,  # Convert to numpy if necessary
    "labels": [labels_dict.get(ytid, None) for ytid in audio_filenames]  # Get corresponding labels from caption
}

# Save to file
torch.save(embedding_dict, "../data/audio_embeddings_with_labels.pt")
print(f"Embedding dictionary saved to ../data/audio_embeddings_with_labels.pt")

if CLOUD_DATA:
    gcs_output_file = "audio_embeddings_with_labels.pt"  # Desired filename in GCS
    lob = bucket.blob(gcs_output_file)
    torch.save(embedding_dict, "../data/audio_embeddings_with_labels.pt")

    # Upload the temporary file to GCS
    blob.upload_from_filename("../data/audio_embeddings_with_labels.pt")

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
print(f"Number of filenames: {len(embedding_dict['filenames'])}")
print(f"Number of labels: {len(embedding_dict['labels'])}")

# Print the first 5 filenames, embeddings, and labels for verification
print("\nFirst 5 entries in the embedding dictionary:")
for i in range(min(5, len(embedding_dict['embeddings']))):
    print(f"Filename: {embedding_dict['filenames'][i]}")
    print(f"Label: {embedding_dict['labels'][i]}")
    print(f"Embedding shape: {embedding_dict['embeddings'][i].shape}")  # Shape of the embedding
    print("-" * 40)  # Separator for clarity