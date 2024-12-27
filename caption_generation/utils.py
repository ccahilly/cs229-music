import argparse
import torch
import shutil
import os

def delete_local_copy(local_path):
    """
    Delete a local file or directory if it exists.
    Args:
        local_path (str): The path to the local file or directory to delete.
    """
    if os.path.isfile(local_path):
        os.remove(local_path)
    elif os.path.isdir(local_path):
        shutil.rmtree(local_path)

def upload_to_gcs(local_path, gcs_path, bucket, delete_locally=True):
    """
    Upload a local file or directory to Google Cloud Storage.
    Args:
        local_path (str): The local file/directory path.
        gcs_path (str): The target GCS path.
    """
    if os.path.isfile(local_path):
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        if delete_locally:
            delete_local_copy(local_path)  # Delete the local file
    elif os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                blob = bucket.blob(os.path.join(gcs_path, relative_path))
                blob.upload_from_filename(local_file_path)
        if delete_locally:
            delete_local_copy(local_path)
    else:
        raise ValueError("Invalid local path: Must be a file or directory")

def download_from_gcs(gcs_path, local_path, bucket):
    """
    Download a file or directory from GCS to a local path.
    """
    # Retrieve blobs as a list to avoid iterator issues
    blobs = list(bucket.list_blobs(prefix=gcs_path))
    if not blobs:  # Check if anything exists under the prefix
        raise ValueError(f"The GCS path '{gcs_path}' does not exist in the bucket.")

    for blob in blobs:
        # Construct the relative path for the local filesystem
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        # print(f"Downloaded {blob.name} to {local_file_path}.")
    
    print(f"Done downloading from {gcs_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning caption generation model.")
    
    # Adding arguments for epochs, last_epoch, and frozen status
    parser.add_argument('--embed_model', type=str, default="clap", help="clap, mert or wav2vec2.")
    parser.add_argument('--frozen', type=bool, default=False, help="Set whether to freeze the embedding model (True/False).")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train the model.")
    parser.add_argument('--last_epoch', type=int, default=0, help="The last epoch used for checkpointing.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
    
    return parser.parse_args()

# Saving model and optimizer checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

# Loading model and optimizer checkpoint
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filename}")
    return model, optimizer, epoch, loss