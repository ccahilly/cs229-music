import random
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import torch
from scipy.io import wavfile
import os
import pandas as pd
import evaluate
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from speecht5_train import SAMPLE_RATE, SpeechDataset, NORMALIZING_INPUT

wer = evaluate.load("wer")

# Paths
split_save_path = "../data/splits"
model_path = "../models/speecht5-model-e15"  # Path to the saved model

# Load the preprocessed data splits
train_data = pd.read_csv(os.path.join(split_save_path, "train.csv"))
val_data = pd.read_csv(os.path.join(split_save_path, "val.csv"))
test_data = pd.read_csv(os.path.join(split_save_path, "test.csv"))

# Print the number of examples in each dataset
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(val_data)}")
print(f"Number of test examples: {len(test_data)}")

def check_overlap(train_data, val_data, test_data):
    """
    Check if there are overlapping datapoints between train, val, and test datasets.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Test dataset.
    
    Returns:
        None. Prints the results of the overlap check.
    """
    # Convert file paths to sets
    train_set = set(train_data["file_path"])
    val_set = set(val_data["file_path"])
    test_set = set(test_data["file_path"])
    
    # Check for overlaps
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    # Print results
    if train_val_overlap:
        print(f"Overlap found between train and val: {len(train_val_overlap)} examples")
    else:
        print("No overlap between train and val.")
    
    if train_test_overlap:
        print(f"Overlap found between train and test: {len(train_test_overlap)} examples")
    else:
        print("No overlap between train and test.")
    
    if val_test_overlap:
        print(f"Overlap found between val and test: {len(val_test_overlap)} examples")
    else:
        print("No overlap between val and test.")

check_overlap(train_data, val_data, test_data)

# Load the processor and model
processor = SpeechT5Processor.from_pretrained(model_path)
model = SpeechT5ForSpeechToText.from_pretrained(model_path)

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

# Prepare DataLoader (though not needed for inference, but we'll use it to sample random examples)
train_dataset = SpeechDataset(train_data, processor)
val_dataset = SpeechDataset(val_data, processor)
test_dataset = SpeechDataset(test_data, processor)

def run_inference(dataset, dataset_name, n_samples=3):
    model.eval()  # Set model to evaluation mode
    total_wer = 0
    random_indices = random.sample(range(len(dataset)), n_samples)
    s = ""

    for idx in tqdm(random_indices, desc=f"Processing {dataset_name}"):
        sample = dataset[idx]
        input_values = sample["input_values"].unsqueeze(0).to(device)
        true_caption = sample["labels"]

        # Generate predictions
        with torch.no_grad():
            generated_ids = model.generate(input_values=input_values)
            predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Compute WER
        decoded_references = [true_caption]
        total_wer += wer.compute(predictions=predicted_texts, references=decoded_references)

        # Log random samples
        s += f"\nSample {idx + 1} from {dataset_name}:\n"
        s += f"True Caption: {true_caption}\n"
        s += f"Generated Caption: {predicted_texts[0]}\n"
        s += "-" * 50
        s += "\n"

    print(s)

    # Calculate average WER for sampled data
    avg_wer = total_wer / n_samples
    print(f"\nAverage WER for {dataset_name}: {avg_wer:.4f}")

# Compute WER for train, val, and test datasets
# print("Processing Train Dataset:")
# run_inference(train_dataset, "Train")

# print("\nProcessing Validation Dataset:")
# run_inference(val_dataset, "Validation")

print("\nProcessing Test Dataset:")
run_inference(test_dataset, "Test")