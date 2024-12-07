#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, Wav2Vec2FeatureExtractor, T5Tokenizer, T5ForConditionalGeneration
import torchaudio.transforms as T
from datasets import load_dataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load Pre-trained Models
mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
decoder = T5ForConditionalGeneration.from_pretrained("t5-small")

# Freeze MERT encoder
for param in mert.parameters():
    param.requires_grad = False

# Aggregator for MERT outputs (weighted average over layers)
aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)


# In[33]:


import torchaudio
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, file_paths, descriptions, processor, resample_rate):
        self.file_paths = file_paths
        self.descriptions = descriptions
        self.processor = processor
        self.resample_rate = resample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        waveform, sample_rate = torchaudio.load(file_path)
    
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # print("Initial shape")
        # print(waveform.shape)
        # print(sample_rate)

        if sample_rate != self.resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)
            waveform = resampler(waveform)
        # print(self.resample_rate)
        waveform = torch.nn.functional.pad(waveform, (0, 240000 - len(waveform[0])), mode="constant", value=0)
        # print(waveform.shape)

        audio_input = self.processor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, return_tensors="pt")

        text = self.descriptions[idx]
        # print(audio_input["input_values"].shape)

        return {"audio_input": audio_input, "text": text}


# In[34]:


class AudioToTextModel(nn.Module):
    def __init__(self, encoder, decoder, aggregator):
        super(AudioToTextModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.aggregator = aggregator
        self.projection = nn.Linear(1024, 512)  # 1024 -> 512

    def forward(self, audio_input, decoder_input_ids=None, labels=None):
#         print(audio_input["input_values"].size())
        with torch.no_grad():
            outputs = self.encoder(**audio_input, output_hidden_states=True)
#             print("Last output hidden")
#             print(outputs.last_hidden_state.shape)
            final_layer_hidden = outputs.last_hidden_state
        
#         print("After encoder")
#         print(final_layer_hidden.shape)
        time_reduced_hidden = final_layer_hidden[:, :512, :]
#         print("Reduced time")
#         print(time_reduced_hidden.shape)
        
        projected_embeddings = self.projection(time_reduced_hidden) 
#         print("After projection")
#         print(projected_embeddings.size())
        
#         seq_length = decoder_input_ids.size(1)
#         projected_embeddings = projected_embeddings.unsqueeze(1) #.expand(-1, seq_length, -1)
        # print("Expanded projected embeddings")
        # print(projected_embeddings.size())
        
        # print("Decoder input ids size")
        # print(decoder_input_ids.size())
        
        # print(labels.size())
#         print(projected_embeddings.shape)
        decoder_outputs = self.decoder(
            inputs_embeds=projected_embeddings,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        return decoder_outputs
        
    def generate(self, audio_input, max_length=50):
        """
        Generates text from audio input using the decoder's generate method.
        
        Args:
            audio_input: Preprocessed audio input.
            max_length: Maximum length of the output sequence.
            num_beams: Number of beams for beam search (default: 1 for greedy decoding).

        Returns:
            Generated token IDs.
        """
        with torch.no_grad():
            outputs = self.encoder(**audio_input, output_hidden_states=True)
#             print("Last output hidden")
#             print(outputs.last_hidden_state.shape)
            final_layer_hidden = outputs.last_hidden_state
        
#         print("After encoder")
#         print(final_layer_hidden.shape)
        time_reduced_hidden = final_layer_hidden[:, :512, :]
#         print("Reduced time")
#         print(time_reduced_hidden.shape)
        
        projected_embeddings = self.projection(time_reduced_hidden)
        print("Projected embeddings shape")
        print(projected_embeddings.shape)
        
        print(projected_embeddings)

        # Use the decoder's `generate` method
        generated_ids = self.decoder.generate(
            inputs_embeds=projected_embeddings,
            max_length=max_length,
        )
        return generated_ids


# In[35]:


# Initialize Model
model = AudioToTextModel(encoder=mert, decoder=decoder, aggregator=aggregator).to(device)

# Optimizer (only train decoder and aggregator)
optimizer = torch.optim.AdamW([
    {"params": model.decoder.parameters(), "lr": 5e-5},
    {"params": model.aggregator.parameters(), "lr": 5e-5},
])


# In[36]:


def collate_fn(batch):
    audio_inputs = {key: torch.cat([item["audio_input"][key] for item in batch], dim=0) for key in batch[0]["audio_input"]}
    captions = [item["text"] for item in batch]

    # Tokenize captions
    tokenized = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
    decoder_input_ids = tokenized.attention_mask
    labels = tokenized.input_ids
    

    return audio_inputs, decoder_input_ids, labels


# In[37]:


import pandas as pd
to_exclude = ["lwdDm3UO5WM", "sETUDPPoDuo", "W58kioYp1Ms"]
data_path = "data/wav_files/wav-48"
train_data = pd.read_csv("train_labels.csv")[["ytid", "caption"]]
train_data = train_data[~train_data['ytid'].isin(to_exclude)]
train_data["ytid"] = [f"{data_path}/{filename}.wav" for filename in train_data["ytid"]]

test_data = pd.read_csv("test_labels.csv")[["ytid", "caption"]]
test_data = test_data[~test_data['ytid'].isin(to_exclude)]
test_data["ytid"] = [f"{data_path}/{filename}.wav" for filename in test_data["ytid"]]


# In[38]:


df = pd.read_csv("train_labels.csv")


# In[39]:


len(train_data)


# In[40]:


from torch.utils.data import DataLoader


resample_rate = processor.sampling_rate
dataset = MusicDataset(list(train_data["ytid"]), list(train_data["caption"]), processor, resample_rate)
test_dataset = MusicDataset(list(test_data["ytid"]), list(test_data["caption"]), processor, resample_rate)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)


# In[41]:


# Training Loop
num_epochs = 3
losses = []
for epoch in range(num_epochs):
    for batch in dataloader:
        
        # Move inputs and labels to GPU
        audio_inputs = {key: value.to(device) for key, value in batch[0].items()}
        decoder_input_ids = batch[1].to(device)
        labels = batch[2].to(device)
        
        # print("Audio input device:", {key: val.device for key, val in audio_inputs.items()})
        # print("Decoder input IDs device:", decoder_input_ids.device)
        # print("Labels device:", labels.device)
        # print("Model device:", next(model.parameters()).device)


        optimizer.zero_grad()

        # Forward pass
        outputs = model(audio_inputs, decoder_input_ids=decoder_input_ids, labels=labels)

        # Compute loss and backpropagate
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(f"Epoch: {epoch}, Loss: {loss.item()}")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script config_template.ipynb')


# In[ ]:


print(torch.__version__)


# In[ ]:


def preprocess_audio(file_path, resample_rate=24000, max_length=240000):
    """Preprocesses the audio file: loads, converts to mono, resamples, and pads/trims."""
    waveform, sample_rate = torchaudio.load(file_path)
        
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # print("Initial shape")
    # print(waveform.shape)
    # print(sample_rate)

    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resampler(waveform)
    waveform = torch.nn.functional.pad(waveform, (0, 240000 - len(waveform[0])), mode="constant", value=0)
    audio_input = processor(waveform.squeeze().numpy(), sampling_rate=24000, return_tensors="pt")
    return audio_input

def inference(audio_file_path):
    """Takes an audio file path, preprocesses it, and outputs the generated text."""
    # Preprocess the audio
    audio_input = preprocess_audio(audio_file_path).to(device)
    print(audio_input)
#     print(model(audio_input))

    # Generate text
#     model.decoder.eval()
#     print(model(audio_input))
    
    generated_tokens = model.generate(audio_input)  # Adjust max_length as needed
    print(generated_tokens)

    # Decode the generated tokens to text
    predicted_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return predicted_text


# In[ ]:


index = 11
print(train_data["ytid"][index])
res = inference(train_data["ytid"][index])
res


# In[ ]:


decoder2 = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)


# In[ ]:


sample_input = torch.randn(1, 512, 512).to(device)  # Simulated embeddings
out_tokens = model.decoder.generate(inputs_embeds=sample_input)
out_tokens


# In[ ]:


tokenizer.decode(out_tokens[0], skip_special_tokens=True)


# In[ ]:


torch.save(model.state_dict(), "mert_t5_a2t.pth")


# In[ ]:




