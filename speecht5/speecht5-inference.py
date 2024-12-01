import random
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import torch
from scipy.io import wavfile
import os
import pandas as pd
import evaluate
from tqdm import tqdm

wer = evaluate.load("wer")

# Paths
split_save_path = "../data/splits"
model_path = "../models/speecht5-model-e15"  # Path to the saved model

# Load the preprocessed data splits
train_data = pd.read_csv(os.path.join(split_save_path, "train.csv"))
val_data = pd.read_csv(os.path.join(split_save_path, "val.csv"))
test_data = pd.read_csv(os.path.join(split_save_path, "test.csv"))

# Load the processor and model
processor = SpeechT5Processor.from_pretrained(model_path)
model = SpeechT5ForSpeechToText.from_pretrained(model_path)

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# Prepare the dataset class
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data, processor, audio_dir="../data/wav"):
        self.data = data
        self.processor = processor
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the audio file
        audio_path = self.data.iloc[idx]["file_path"]
        
        # Load the audio file using scipy (or librosa if you prefer)
        _, audio = wavfile.read(audio_path)
        
        # Check if the audio is stereo (2 channels) and convert it to mono if necessary
        if audio.ndim == 2:  # Stereo audio (2 channels)
            audio = audio.mean(axis=1)  # Convert stereo to mono by averaging the channels
            
        # Use the processor to convert audio into input values (with proper sample rate)
        audio_input = self.processor(audio=audio, sampling_rate=16000, return_tensors="pt")

        # Get the text caption (this is used as the target label for the model)
        caption = self.data.iloc[idx]["caption"]

        # Return processed audio and corresponding caption (label)
        return {"input_values": audio_input["input_values"].squeeze(), "labels": caption}

# Prepare DataLoader (though not needed for inference, but we'll use it to sample random examples)
train_dataset = SpeechDataset(train_data, processor)
val_dataset = SpeechDataset(val_data, processor)
test_dataset = SpeechDataset(test_data, processor)

# Function to run inference and print results
def run_inference(dataset, n_samples=3):
    model.eval()  # Set model to evaluation mode
    total_wer = 0
    random_indices = random.sample(range(len(dataset)), n_samples)

    for idx in random_indices:
        # Load the audio file and caption from the dataset
        sample = dataset[idx]
        input_values = sample["input_values"].unsqueeze(0).to(device)  # Add batch dimension and move to device
        true_caption = sample["labels"]
        
        # Tokenize the labels (captions)
        labels = processor(text_target=true_caption, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(input_values=input_values, labels=labels)  # Using labels for supervised inference

        # Get the predicted text
        predictions = outputs.logits.argmax(dim=-1)  # Get the predicted tokens
        predicted_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Decode the true labels to text
        decoded_references = processor.batch_decode(labels, skip_special_tokens=True)

        # Compute WER (Word Error Rate)
        total_wer += wer.compute(predictions=predicted_texts, references=decoded_references)

        # Print results for the sample
        print(f"True Caption: {true_caption}")
        print(f"Generated Caption: {predicted_texts[0]}")
        print("-" * 50)
    
    avg_wer = total_wer / n_samples
    print(f"Average WER: {avg_wer:.4f}")

# Run inference for random samples from the train, val, and test sets
print("Inference on 3 random samples from the Train set:")
run_inference(train_dataset, n_samples=3)

print("\nInference on 3 random samples from the Validation set:")
run_inference(val_dataset, n_samples=3)

print("\nInference on 3 random samples from the Test set:")
run_inference(test_dataset, n_samples=3)
