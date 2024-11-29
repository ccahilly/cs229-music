import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Split data
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Save splits
os.makedirs(split_save_path, exist_ok=True)
train_data.to_csv(os.path.join(split_save_path, "train.csv"), index=False)
val_data.to_csv(os.path.join(split_save_path, "val.csv"), index=False)
test_data.to_csv(os.path.join(split_save_path, "test.csv"), index=False)

print("Data splits saved!")