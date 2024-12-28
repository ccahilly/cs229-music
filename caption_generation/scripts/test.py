# Run from caption_generation directory with:
# python -m scripts.test

import torch
from torch.utils.data import DataLoader
from models import ClapT5Model
from models import MertT5Model
from models import Wav2Vec2T5Model
from transformers import T5Tokenizer, Wav2Vec2Processor, AutoProcessor
from utils import load_checkpoint, evaluate, parse_args, calculate_bert_similarity
from dataset import ClapAudioCaptionDataset, MertAudioCaptionDataset, Wav2Vec2AudioCaptionDataset  # Adjust if needed

if __name__ == "__main__":
    # Setup & hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_GCP = False
    test_data_path = "../data/splits/test.csv"

    print("Device:", DEVICE)

    # Load model configuration
    args = parse_args()
    EMBED_MODEL = args.embedding
    FROZEN = args.frozen
    LAST_EPOCH = args.last_epoch
    print(f"Training configuration: Embed Model = {EMBED_MODEL}, Frozen = {FROZEN}, Epoch = {LAST_EPOCH}")

    model_save_path = f"checkpoints/{EMBED_MODEL}_t5_"
    if FROZEN:
        model_save_path += "frozen"
    else:
        model_save_path += "unfrozen"

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    if EMBED_MODEL == "clap":
        BATCH_SIZE = 8
        audio_processor = AutoProcessor.from_pretrained("laion/larger_clap_music")
        model = ClapT5Model(DEVICE, frozen=FROZEN)
        AudioCaptionDataset = ClapAudioCaptionDataset
    elif EMBED_MODEL == "mert":
        BATCH_SIZE = 4
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = MertT5Model(DEVICE, frozen=FROZEN, batch_size=BATCH_SIZE)
        AudioCaptionDataset = MertAudioCaptionDataset
    elif EMBED_MODEL == "wav2vec2":
        BATCH_SIZE = 8
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2T5Model(DEVICE, frozen=FROZEN)
        AudioCaptionDataset = Wav2Vec2AudioCaptionDataset
    else:
        raise ValueError("Invalid embedding model specified.")

    # Load dataset
    test_dataset = AudioCaptionDataset(test_data_path, audio_processor, t5_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Load checkpoint (if available)
    model, _, _, _ = load_checkpoint(model, None, model_save_path + f"/checkpoint{LAST_EPOCH}.pth")  # Adjust for correct checkpoint file

    # Evaluation on test dataset
    test_loss, predictions, true_labels = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Print 8 arbitrary example outputs and true values along with BERT similarity
    for i in range(8):
        true_caption = t5_tokenizer.decode(true_labels[i], skip_special_tokens=True)
        predicted_caption = t5_tokenizer.decode(predictions[i], skip_special_tokens=True)
        
        # Calculate BERT similarity score
        bert_sim_score = calculate_bert_similarity(true_caption, predicted_caption)

        print(f"Example {i+1}:")
        print(f"True Caption: {true_caption}")
        print(f"Predicted Caption: {predicted_caption}")
        print(f"BERT Similarity Score: {bert_sim_score:.4f}")
        print("-" * 50)

    # Calculate BERT similarity for all test examples
    all_bert_similarities = []
    for i in range(len(true_labels)):
        true_caption = t5_tokenizer.decode(true_labels[i], skip_special_tokens=True)
        predicted_caption = t5_tokenizer.decode(predictions[i], skip_special_tokens=True)
        
        # Calculate BERT similarity score for each example
        bert_similarity = calculate_bert_similarity(true_caption, predicted_caption)
        all_bert_similarities.append(bert_similarity)

    # Calculate and print the overall average BERT similarity for all test examples
    overall_average_bert_sim = sum(all_bert_similarities) / len(all_bert_similarities)
    print(f"\nOverall Average BERT Similarity for all test examples: {overall_average_bert_sim:.4f}")