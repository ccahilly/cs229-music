# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_from_disk
from prep_data_for_cos_sim import data_dir 
from scipy.spatial.distance import cosine
import numpy as np

# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

# load demo audio and set processor
dataset = load_from_disk(data_dir)
all_embeddings = []
for i in range(len(dataset)):
    input_audio = dataset[i]["audio"]["audio_array"]
    # print(np.array(input_audio).shape)
    
    input = processor(input_audio, sampling_rate=processor.sampling_rate, return_tensors="pt")
    # print(input["input_values"].shape)
    with torch.no_grad():
        outputs = model(**input, output_hidden_states=True)

    # take a look at the output shape, there are 13 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    # print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]
    all_embeddings.append(all_layer_hidden_states)

# Function to compute the cosine similarity between two vectors
def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    """
    return 1 - cosine(embedding1.flatten(), embedding2.flatten())

num_layers = all_layer_hidden_states.shape[0]
for i in range(num_layers):
    print(f"Layer {i+1}")
    
    all_cosine_similarities = []
    for j in range(len(all_embeddings)):
        for k in range(j + 1, len(all_embeddings)):
            print(compute_cosine_similarity(all_embeddings[j][i], all_embeddings[k][i]))
            all_cosine_similarities.append(compute_cosine_similarity(all_embeddings[j][i], all_embeddings[k][i]))
# all_embeddings = torch.stack(all_embeddings)
# print(all_embeddings.shape)
# print(all_embeddings.unsqueeze(0).shape)

# # for utterance level classification tasks, you can simply reduce the representation in time
# time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
# print(time_reduced_hidden_states.shape) # [13, 768]

# you can even use a learnable weighted average representation
# aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
# weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
# print(weighted_avg_hidden_states.shape) # [768]

# # Combine time steps (749) and features (768) into one dimension
# combined_dim = all_embeddings.view(10, 13, -1)  # [batch_size, layers, time_steps * features]

# # Apply Conv1d for learnable aggregation
# conv_aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)  # Kernel size of 1 keeps the feature dimension intact
# aggregated_embedding = conv_aggregator(combined_dim)  # [batch_size, 1, time_steps * features]

# # Uncombine the last dimension back into time_steps and features
# aggregated_embedding = aggregated_embedding.view(10, 749, 768)  # [batch_size, time_steps, features]
# print(aggregated_embedding.shape)