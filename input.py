from main import tweets, stop_words, unigram_counter, context_counter, frequent_unigrams, build_word_vector
# from sklearn.utils.extmath import randomized_svd
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
# from collections import Counter
import numpy as np
from time import perf_counter

def get_cosine_similarity(word1_vector: dict[str, float], word2_vector: dict[str, float]) -> float:
    # Convert dictionaries to numpy arrays
    vec1 = np.array([word1_vector.get(word) for word in word1_vector.keys()])
    vec2 = np.array([word2_vector.get(word) for word in word2_vector.keys()])
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    
    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    tic = perf_counter()
    print(get_cosine_similarity(word1_vector, word2_vector))
    toc = perf_counter()
    print(f"Time taken: {toc - tic} seconds")

    