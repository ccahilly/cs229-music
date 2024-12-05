import shutil
from google.cloud import storage

# Initialize Google Cloud Storage client
gcs_bucket_name = "musiccaps-wav-16khz"
storage_client = storage.Client()
bucket = storage_client.bucket(gcs_bucket_name)

def upload_to_gcs(local_path, gcs_path):
    """
    Upload a local file or directory to Google Cloud Storage.
    Args:
        local_path (str): The local file/directory path.
        gcs_path (str): The target GCS path.
    """
    if os.path.isfile(local_path):
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        os.remove(local_path)  # Delete the local file
    elif os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                blob = bucket.blob(os.path.join(gcs_path, relative_path))
                blob.upload_from_filename(local_file_path)
        shutil.rmtree(local_path)  # Delete the local directory after upload
    else:
        raise ValueError("Invalid local path: Must be a file or directory")