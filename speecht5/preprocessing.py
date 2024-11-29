import os
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

# Paths
audio_dir = "../data/wav"
metadata_path = "../data/musiccaps-train-data.csv"
split_save_path = "../data/splits"

# Read metadata
metadata = pd.read_csv(metadata_path)

# Ensure relevant columns exist
assert "ytid" in metadata.columns and "caption" in metadata.columns, "Missing required columns!"

# Add full file paths
metadata["file_path"] = metadata["ytid"].apply(lambda x: os.path.join(audio_dir, f"{x}.wav"))

# List of ytids to ignore
ytids_to_ignore = ['W58kioYp1Ms', 'lwdDm3UO5WM', 'sETUDPPoDuo']

# Function to get the sample rate of an audio file
def get_sample_rate(file_path):
    try:
        sample_rate, _ = wavfile.read(file_path)
        return sample_rate
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Filter out entries where the corresponding .wav file doesn't exist or has a sample rate < 16000, 
# except for the ytids in ytids_to_ignore
metadata = metadata[metadata["ytid"].apply(lambda x: x not in ytids_to_ignore)]  # Exclude ytids to ignore
metadata = metadata[metadata["file_path"].apply(lambda x: os.path.exists(x) and get_sample_rate(x) >= 16000)]

# Split data into train, validation, and test sets
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save splits to CSV files
os.makedirs(split_save_path, exist_ok=True)
train_data.to_csv(os.path.join(split_save_path, "train.csv"), index=False)
val_data.to_csv(os.path.join(split_save_path, "val.csv"), index=False)
test_data.to_csv(os.path.join(split_save_path, "test.csv"), index=False)

print("Data splits saved!")