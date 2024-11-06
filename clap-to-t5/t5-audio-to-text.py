# General Approach
# Create a projection layer to transform the (512,) embeddings to the T5 input dimension
# and then set up a model architecture that feeds these transformed embeddings into T5. 

# Starting with T5-small for prototyping
# Num params small / base / large / 3B / 11B: 60M / 220M / 770M / 3B / 11B

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

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
        
        # Generate outputs with T5
        outputs = self.t5(
            inputs_embeds=projected_embeddings,
            labels=labels
        )
        return outputs

# Load the training data
train_data = torch.load('../data/train_data.pt')

# Extract the embeddings and labels from the loaded data
train_embeddings = torch.tensor(train_data["embeddings"])
train_labels = train_data["labels"]

# Prepare DataLoader with your train, validation splits
train_dataset = TensorDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))  # Adapt to your data structure
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model and tokenizer
model = AudioToTextModel(projection_dim=768)  # or use a larger dimension for more complexity
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Or another T5 model depending on your choice

# Set the model to training mode
model.train()

# Set the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

num_epochs = 5  # You can adjust this depending on your dataset and model size

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        audio_embeddings, labels = batch
        
        # Tokenize the labels for T5
        tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt").input_ids
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(audio_embeddings, labels=tokenized_labels)
        
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