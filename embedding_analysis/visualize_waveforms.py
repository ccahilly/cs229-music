import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel, Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load audio file
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# Visualize waveform function
def plot_waveform(waveform, sample_rate=16000, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.savefig("waveform.png")

# CLAP Processor and Model
def process_clap(file_path):
    processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
    model = AutoModel.from_pretrained("laion/larger_clap_music").to(DEVICE)
    
    waveform, sample_rate = load_audio(file_path)
    
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
    embeddings = model(**inputs).last_hidden_state  # Get the embedding
    plot_waveform(waveform, sample_rate, "CLAP Audio Embedding")

# Wav2Vec2 Processor and Model
def process_wav2vec(file_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    
    waveform, sample_rate = load_audio(file_path)
    
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
    embeddings = wav2vec_model(**inputs).last_hidden_state  # Get the embedding
    plot_waveform(waveform, sample_rate, "Wav2Vec2 Audio Embedding")

# MERT Processor and Model
def process_mert(file_path):
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
    
    waveform, sample_rate = load_audio(file_path)
    
    inputs = mert_processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(DEVICE)
    embeddings = mert_model(**inputs).last_hidden_state  # Get the embedding
    plot_waveform(waveform, sample_rate, "MERT Audio Embedding")

# Load CSV and process file paths
def process_csv_and_audio(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Iterate over rows and process each audio file
    for index, row in df.iterrows():
        file_path = row['file_path']  # Get file path from 'file_path' column
        caption = row['caption']      # Get caption from 'caption' column
        
        print(f"Processing file: {file_path}, Caption: {caption}")
        
        # Process with CLAP
        process_clap(file_path)
        
        # Process with Wav2Vec2
        process_wav2vec(file_path)
        
        # Process with MERT
        process_mert(file_path)

# Example usage with the CSV file path
csv_file_path = "../data/splits/test.csv"
process_csv_and_audio(csv_file_path)