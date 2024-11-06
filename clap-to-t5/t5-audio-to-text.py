import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AudioToTextModel(nn.Module):
    def __init__(self):
        super(AudioToTextModel, self).__init__()
        # Initialize T5 model and tokenizer
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def forward(self, audio_embeddings, labels=None):
        # Ensure correct shape for inputs_embeds: (batch_size, seq_length, embedding_dim)
        # T5 expects the shape (batch_size, seq_length, embedding_dim)
        projected_embeddings = audio_embeddings.unsqueeze(1)  # Add seq_length dimension (usually 1 for this case)

        # Generate outputs with T5
        outputs = self.t5(
            inputs_embeds=projected_embeddings,
            labels=labels
        )
        return outputs

# Load the training data
train_data = torch.load('../data/train_data.pt')

train_embeddings = torch.tensor(np.array(train_data["embeddings"])).to(device)  # Move to GPU
train_labels = [str(label) for label in train_data["labels"]]

# Ensure all labels are strings
for label in train_labels:
    if label is None or not isinstance(label, str):
        print("Label has an error or is not a string")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Tokenize the labels (convert them into token IDs) just once
tokenized_labels = tokenizer(train_labels, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)  # Move to GPU

# Create a DataLoader for your train data
train_dataset = TensorDataset(train_embeddings, tokenized_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model and tokenizer
model = AudioToTextModel().to(device)  # Move the model to GPU
model.train()  # Set the model to training mode

# Set the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

num_epochs = 10  # You can adjust this depending on your dataset and model size

# List to store the loss for each epoch
epoch_losses = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        audio_embeddings, labels = batch
        
        # Move data to GPU
        audio_embeddings = audio_embeddings.to(device)
        labels = labels.to(device)
        
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

        if i % 62 == 0:
            print(f"Done with first {8 * i} examples")

    # Calculate and print the loss for this epoch
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)  # Store the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss}")

# Plotting the training loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', color='b', label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Save the plot as a .jpg file
plt.savefig('audio_to_text_training_loss.jpg', format='jpg')

# Optionally show the plot
plt.show()

# Save the trained model
torch.save(model.state_dict(), '../models/trained_audio_to_text_model.pth')

# save to google cloud
bucket_name = "musiccaps-wav-16khz"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("trained_audio_to_text_model.pth")

# # Upload the temporary file to GCS
# blob.upload_from_filename('../models/trained_audio_to_text_model.pth')
# print(f"Embedding dictionary saved to GCS")