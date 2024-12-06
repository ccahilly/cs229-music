import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle

def load_generated_captions(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def average_pairwise_sim(embeddings):
    pairwise_similarities = cosine_similarity(embeddings)

    # Get the number of embeddings
    num_embeddings = len(embeddings)

    # Extract the upper triangle of the similarity matrix without the diagonal
    upper_triangle_indices = np.triu_indices(num_embeddings, k=1)
    pairwise_values = pairwise_similarities[upper_triangle_indices]

    # Calculate the average pairwise similarity
    average_pairwise_similarity = np.mean(pairwise_values)
    return average_pairwise_similarity

def calculate_bert_scores(true, pred):
    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Split the strings into tokens
    scores = {}
    scores["bleu"] = []
    scores["bert_sim"] = []
    
    for i in range(len(true)):
        reference = true[i].split()
        candidate = pred[i].split()

        # Calculate the BLEU score
        bleu_score = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
        scores["bleu"].append(bleu_score)
        
        # Initializing the Sentence Transformer model using BERT with mean-tokens pooling
        
        # Encoding the sentences to obtain their embeddings
        sentence_embeddings = bert_model.encode([true[i], pred[i]])

        # Calculating the cosine similarity between the first sentence embedding and the rest of the embeddings
        # The result will be a list of similarity scores between the first sentence and each of the other sentences
        similarity_score = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0]
        
        scores["bert_sim"].append(similarity_score)
    
    return scores

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score for a single reference and candidate pair.
    
    :param reference: The ground truth text (as a list of tokens).
    :param candidate: The generated text (as a list of tokens).
    :return: BLEU score (float).
    """
    smoothing = SmoothingFunction().method1
    
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)

def calculate_scores(captions):
    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    scores = {"bleu": [], "bert_sim": []}
    
    i = 0
    for true_caption, generated_caption in captions:
        # BLEU score
        reference = true_caption.split()
        candidate = generated_caption.split()
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothing)
        scores["bleu"].append(bleu_score)
        
        # BERT similarity
        sentence_embeddings = bert_model.encode([true_caption, generated_caption])
        similarity_score = cosine_similarity(
            [sentence_embeddings[0]], [sentence_embeddings[1]]
        )[0][0]
        scores["bert_sim"].append(similarity_score)

        if i % 100 == 0:
            print(f"Done with {i+1} of {len(captions)} captions")
        i += 1

    return scores

# Summarize the results
def summarize_scores(scores):
    bleu_avg = np.mean(scores["bleu"])
    bert_sim_avg = np.mean(scores["bert_sim"])
    return {"bleu_avg": bleu_avg, "bert_sim_avg": bert_sim_avg}

# Main function
def main():
    frozen_file = "wav2vec-to-t5/generated_captions_frozen_e15.pkl"
    unfrozen_file = "wav2vec-to-t5/generated_captions_unfrozen_e14.pkl"

    # Load generated captions
    frozen_captions = load_generated_captions(frozen_file)
    unfrozen_captions = load_generated_captions(unfrozen_file)
    
    # Calculate scores
    print("Calculating scores for frozen model...")
    frozen_scores = calculate_scores(frozen_captions)
    frozen_summary = summarize_scores(frozen_scores)

    print("Calculating scores for unfrozen model...")
    unfrozen_scores = calculate_scores(unfrozen_captions)
    unfrozen_summary = summarize_scores(unfrozen_scores)

    # Print summaries
    print("\nFrozen Model Results:")
    print(f"Average BLEU Score: {frozen_summary['bleu_avg']:.4f}")
    print(f"Average BERT Similarity: {frozen_summary['bert_sim_avg']:.4f}")

    print("\nUnfrozen Model Results:")
    print(f"Average BLEU Score: {unfrozen_summary['bleu_avg']:.4f}")
    print(f"Average BERT Similarity: {unfrozen_summary['bert_sim_avg']:.4f}")

if __name__ == "__main__":
    main()