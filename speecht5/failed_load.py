import os
import pandas as pd
from scipy.io import wavfile

# Paths
split_dir = "../data/splits"  # directory containing train.csv, val.csv, test.csv
audio_dir = "../data/wav"     # directory containing the audio files

# Function to load audio and check for errors (including length check)
def check_audio_file(file_path):
    try:
        # Try loading the audio file
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Check if file is empty or has a length < 160000 samples
        if len(audio_data) == 0 or len(audio_data) < 160000:
            print(len(audio_data))
            return False
        return True
    except Exception as e:
        # If any error occurs (e.g., file not found, corrupted file), return False
        return False

# Function to load CSV and check all files
def check_split_files(split_file):
    # Read metadata CSV
    metadata = pd.read_csv(split_file)
    
    # Ensure that the "ytid" and "file_path" columns exist
    if "ytid" not in metadata.columns:
        print(f"Missing 'ytid' column in {split_file}")
        return []
    
    failed_files = []
    for _, row in metadata.iterrows():
        file_path = os.path.join(audio_dir, f"{row['ytid']}.wav")
        if not check_audio_file(file_path):
            failed_files.append(row['ytid'])
    
    return failed_files

# Check all splits
failed_train_files = check_split_files(os.path.join(split_dir, "train.csv"))
failed_val_files = check_split_files(os.path.join(split_dir, "val.csv"))
failed_test_files = check_split_files(os.path.join(split_dir, "test.csv"))

# Output results
if failed_train_files:
    print(f"Failed files in train split: {failed_train_files}")
else:
    print("All files in train split loaded successfully.")

if failed_val_files:
    print(f"Failed files in validation split: {failed_val_files}")
else:
    print("All files in validation split loaded successfully.")

if failed_test_files:
    print(f"Failed files in test split: {failed_test_files}")
else:
    print("All files in test split loaded successfully.")
