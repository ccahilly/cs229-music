import torch
from sklearn.model_selection import train_test_split

# Load your embedding dictionary
embedding_data = torch.load('../data/audio_embeddings_with_labels.pt')

# Unpack data for easier handling
filenames = embedding_data["filenames"]
embeddings = embedding_data["embeddings"]
labels = embedding_data["labels"]

# Split the data into train, validation, and test sets
# Train is 70%, validation is 15%, and test is 15%
train_files, temp_files, train_embeddings, temp_embeddings, train_labels, temp_labels = train_test_split(
    filenames, embeddings, labels, test_size=0.3, random_state=42
)
val_files, test_files, val_embeddings, test_embeddings, val_labels, test_labels = train_test_split(
    temp_files, temp_embeddings, temp_labels, test_size=0.5, random_state=42
)

# Save splits for easy access
train_data = {"filenames": train_files, "embeddings": train_embeddings, "labels": train_labels}
val_data = {"filenames": val_files, "embeddings": val_embeddings, "labels": val_labels}
test_data = {"filenames": test_files, "embeddings": test_embeddings, "labels": test_labels}

torch.save(train_data, '../data/train_data.pt')
torch.save(val_data, '../data/val_data.pt')
torch.save(test_data, '../data/test_data.pt')


