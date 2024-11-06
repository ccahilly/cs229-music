import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class AudioToTextModel(nn.Module):
    def __init__(self, projection_dim=768):
        super(AudioToTextModel, self).__init__()
        # Initialize T5 model and tokenizer
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small") 
        
        # Add a projection layer to map (512,) -> T5 input size (e.g., 768)
        self.projection = nn.Linear(512, projection_dim)
        
        # Adjust T5 embedding size if different from the original
        self.t5.encoder.embed_tokens = nn.Embedding(self.t5.config.vocab_size, projection_dim)
        
    def forward(self, audio_embeddings, labels=None):
        # Pass through the projection layer
        projected_embeddings = self.projection(audio_embeddings)
    
        # Ensure correct shape for inputs_embeds: (batch_size, seq_length, embedding_dim)
        projected_embeddings = projected_embeddings.unsqueeze(1)  # Add a dimension for sequence length (e.g., 1)

        # Generate outputs with T5
        outputs = self.t5(
            inputs_embeds=projected_embeddings,
            labels=labels
        )
        return outputs


# Load the training data
train_data = torch.load('../data/train_data.pt')

train_embeddings = torch.tensor(np.array(train_data["embeddings"]))
train_labels = [str(label) for label in train_data["labels"]]

# print(f"Type of train labels: {type(train_labels)}")
# print(f"Type of first element of train labels: {type(train_labels[0])}")

# Ensure all labels are strings
for label in train_labels:
    if label is None or not isinstance(label, str):
        print("Label has an error or is not a string")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize the labels (convert them into token IDs) just once
tokenized_labels = tokenizer(train_labels, padding=True, truncation=True, return_tensors="pt").input_ids

# Create a DataLoader for your train data
train_dataset = TensorDataset(train_embeddings, tokenized_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model and tokenizer
model = AudioToTextModel(projection_dim=768)  # Or use a larger dimension for more complexity
model.train()  # Set the model to training mode

# Set the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

num_epochs = 5  # You can adjust this depending on your dataset and model size

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        audio_embeddings, labels = batch
        
        # No need to re-tokenize the labels here, just use the already tokenized labels
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(audio_embeddings, labels=labels)
        
        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        optimizer.step()

    # Print the loss for this epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss}")

# Save the trained model
torch.save(model.state_dict(), '../data/trained_audio_to_text_model.pth')