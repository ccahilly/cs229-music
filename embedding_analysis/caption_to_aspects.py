import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm  # For progress bars
import numpy as np

# Load data
train_df = pd.read_csv('../data/splits/train.csv')
val_df = pd.read_csv('../data/splits/val.csv')
test_df = pd.read_csv('../data/splits/test.csv')

# Convert aspect_list column to actual lists
train_df['aspect_list'] = train_df['aspect_list'].apply(ast.literal_eval)

# Create and fit the MultiLabelBinarizer on the training data
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['aspect_list'])

# Transform validation and test data using the same mlb
val_df['aspect_list'] = val_df['aspect_list'].apply(ast.literal_eval)
val_df['aspect_list'] = val_df['aspect_list'].apply(
    lambda aspects: [aspect for aspect in aspects if aspect in mlb.classes_]
)
y_val = mlb.transform(val_df['aspect_list'])

test_df['aspect_list'] = test_df['aspect_list'].apply(ast.literal_eval)
test_df['aspect_list'] = test_df['aspect_list'].apply(
    lambda aspects: [aspect for aspect in aspects if aspect in mlb.classes_]
)
y_test = mlb.transform(test_df['aspect_list'])

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class AudioCaptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Prepare DataLoaders
train_dataset = AudioCaptionDataset(train_df['caption'], y_train, tokenizer)
val_dataset = AudioCaptionDataset(val_df['caption'], y_val, tokenizer)
test_dataset = AudioCaptionDataset(test_df['caption'], y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

class AspectPredictionModel(nn.Module):
    def __init__(self, num_labels):
        super(AspectPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()  # Output probabilities for each label

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        probs = self.sigmoid(logits)
        return probs

# Define the device at the start of the script
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model initialization
model = AspectPredictionModel(num_labels=y_train.shape[1])  # 13219 aspects
model = model.to(device)  # Move the model to the device

# Training loop
# Training loop with progress updates
def train_epoch(model, data_loader, optimizer, criterion):
    model = model.train()
    losses = []
    all_preds = []
    all_labels = []
    
    # Create a progress bar
    loop = tqdm(data_loader, leave=True)
    loop.set_description("Training")
    
    for data in loop:
        input_ids = data['input_ids'].to(device)  # Move to device
        attention_mask = data['attention_mask'].to(device)  # Move to device
        labels = data['labels'].to(device)  # Move to device

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        all_preds.append(outputs.detach().cpu().numpy())  # Move back to CPU for evaluation
        all_labels.append(labels.detach().cpu().numpy())  # Move back to CPU for evaluation

        # Update progress bar with current loss
        loop.set_postfix(loss=loss.item())
    
    avg_loss = sum(losses) / len(losses)
    
    # Calculate F1 score (for evaluation)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, (all_preds > 0.5), average='macro')
    
    return avg_loss, f1

# Validation loop
def eval_epoch(model, data_loader, criterion):
    model = model.eval()
    losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)  # Move to device
            attention_mask = data['attention_mask'].to(device)  # Move to device
            labels = data['labels'].to(device)  # Move to device

            outputs = model(input_ids, attention_mask)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            all_preds.append(outputs.detach().cpu().numpy())  # Move back to CPU for evaluation
            all_labels.append(labels.detach().cpu().numpy())  # Move back to CPU for evaluation

    avg_loss = sum(losses) / len(losses)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels, (all_preds > 0.5), average='macro')
    
    return avg_loss, f1

optimizer = Adam(model.parameters(), lr=2e-5)  # Adam optimizer with learning rate 2e-5
criterion = nn.BCELoss()  # Binary Cross-Entropy for multi-label classification

# Training process
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_f1 = eval_epoch(model, val_loader, criterion)
    
    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation F1: {val_f1:.4f}")

# Evaluate on the test dataset
# test_loss, test_f1 = eval_epoch(model, test_loader, criterion)
# print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

def top_n_match_accuracy(model, data_loader):
    model.eval()
    total_matches = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            true_labels = data['labels'].cpu().numpy()  # Move true labels to CPU
            
            # Get model predictions
            outputs = model(input_ids, attention_mask)
            predictions = outputs.detach().cpu().numpy()  # Move predictions to CPU
            
            for i in range(len(true_labels)):
                true_aspects = np.where(true_labels[i] > 0)[0]  # Indices of true aspects
                n = len(true_aspects)
                
                if n == 0:
                    continue  # Skip if there are no true aspects
                
                # Get the top-n predicted aspect indices
                top_n_predictions = predictions[i].argsort()[-n:][::-1]
                
                # Count matches
                matches = len(set(true_aspects) & set(top_n_predictions))
                total_matches += matches  # Match percentage for this example
                total_samples += n
    
    # Average match percentage over all samples
    accuracy = total_matches / total_samples if total_samples > 0 else 0
    return accuracy

# Evaluate on the test dataset
test_accuracy = top_n_match_accuracy(model, test_loader)
print(f"Top-n Match Accuracy: {test_accuracy * 100:.2f}%")
