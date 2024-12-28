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
from tqdm import tqdm 

if __name__ == "__main__":
    # Setup & hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_GCP = False
    test_data_path = "../data/splits/test.csv"
    eval_all = False

    print("Device:", DEVICE)

    # Load model configuration
    args = parse_args()
    EMBED_MODEL = args.embedding
    FROZEN = args.frozen
    LAST_EPOCH = args.last_epoch
    print(f"Configuration: Embed Model = {EMBED_MODEL}, Frozen = {FROZEN}, Epoch = {LAST_EPOCH}")

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

    model.eval()

    # Inference loop
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            # Get model predictions
            predictions = model.inference(batch, t5_tokenizer)
            all_predictions.extend(predictions)

            # Decode true labels to text
            labels = batch["labels"].to(DEVICE)
            true_captions = [t5_tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            all_true_labels.extend(true_captions)  # Add true captions to the list

    # Print or save predictions
    i = 0
    all_bert_similarities = []
    for pred, true in zip(all_predictions, all_true_labels):
        bert_similarity = calculate_bert_similarity(true, pred)
        if i < 8:
            print(f"Predicted: {pred}")
            print(f"True: {true}")
            print(f"Bert Similarity: {bert_similarity:.4f}")
            print("-" * 80)
        i += 1
        all_bert_similarities.append(bert_similarity)
        if not eval_all and i >= 8:
            break

    # Calculate and print the overall average BERT similarity for all test examples
    overall_average_bert_sim = sum(all_bert_similarities) / len(all_bert_similarities)
    print(f"Overall Average BERT Similarity for all test examples: {overall_average_bert_sim:.4f}")