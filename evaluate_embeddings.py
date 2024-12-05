import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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