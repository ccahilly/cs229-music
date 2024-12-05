import torch
from transformers import Wav2Vec2FeatureExtractor, T5Tokenizer, T5ForConditionalGeneration, AutoModel
import numpy as np
from scipy.io import wavfile
import torch.nn as nn
import os
from mert_to_t5_train import AudioCaptionDataset, preprocess_audio, MAX_TOKENS
import pandas as pd

data_dir = "../data/splits"
model_save_path = "../models/fine_tuned_mert_t5_e5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load the fine-tuned model, Wav2Vec2 model, linear layer, and processor
train_data_path = "../data/splits/train.csv"
val_data_path = "../data/splits/val.csv"
test_data_path = "../data/splits/test.csv"

mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_save_path + "/mert")
mert_model = AutoModel.from_pretrained(model_save_path + "/mert").to(DEVICE)

t5_tokenizer = T5Tokenizer.from_pretrained(model_save_path + "/t5")
t5_model = T5ForConditionalGeneration.from_pretrained(model_save_path + "/t5").to(DEVICE)

aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1).to(DEVICE)
aggregator.load_state_dict(torch.load(os.path.join(model_save_path + "/aggregator", "aggregator.pth")))

reduce_layer = nn.Linear(768, t5_model.config.d_model).to(DEVICE)
reduce_layer.load_state_dict(torch.load(os.path.join(model_save_path + "/linear", "reduce_layer.pth")))

def infer(test_example):
    # Extract input data from the test example
    input_values = test_example["inputs"].unsqueeze(0).to(DEVICE)  # Add batch dimension
    # attention_mask = test_example["attention_mask"].unsqueeze(0).to(DEVICE)
    labels = test_example["labels"].unsqueeze(0).to(DEVICE)  # You may not need labels for inference
    # decoder_attention_mask = test_example["decoder_attention_mask"].unsqueeze(0).to(DEVICE)

    # Pass audio through MERT model to get embeddings
    mert_outputs = mert_model(input_values, output_hidden_states=True)
    all_layer_hidden_states = torch.stack(mert_outputs.hidden_states).squeeze()
    combined_dim = all_layer_hidden_states.view(1, 13, -1)  # Adjust for single input

    # Apply the aggregator to learnable aggregation
    aggregated_embedding = aggregator(combined_dim).view(1, 749, 768)  # Shape adjustment

    # Reduce the Wav2Vec2 embeddings to match T5 input dimensions
    reduced_embeddings = reduce_layer(aggregated_embedding)

    # Generate caption using T5 model
    t5_input_ids = t5_tokenizer.encode("Generate caption:", return_tensors="pt").to(DEVICE)
    outputs = t5_model.generate(
        inputs_embeds=reduced_embeddings,
        decoder_input_ids=t5_input_ids,
        max_length=MAX_TOKENS,
        num_beams=5,
        early_stopping=True
    )

    # Decode the output caption
    generated_caption = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Caption:", generated_caption)
    # print("Real caption")
    # print(labels)


if __name__ == "__main__":
    # Instantiate the dataset (for just one test file)
    test_dataset = AudioCaptionDataset(data_dir + "/test.csv", mert_processor, t5_tokenizer)
    infer(test_dataset[0])
    infer(test_dataset[1])
    infer(test_dataset[2])