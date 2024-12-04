# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_from_disk
from prep_data_for_cos_sim import data_dir 
from scipy.spatial.distance import cosine

# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

# load demo audio and set processor
dataset = load_from_disk(data_dir)
all_embeddings = []
for i in range(len(dataset)):
    input_audio = dataset[i]["audio"]["audio_array"]
    
    input = processor(input_audio, sampling_rate=processor.sampling_rate, return_tensors="pt")
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

# # for utterance level classification tasks, you can simply reduce the representation in time
# time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
# print(time_reduced_hidden_states.shape) # [13, 768]

# # you can even use a learnable weighted average representation
# aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
# weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
# print(weighted_avg_hidden_states.shape) # [768]