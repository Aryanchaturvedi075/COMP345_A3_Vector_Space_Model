from sklearn.utils.extmath import randomized_svd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import re


def top_k_unigrams(tweets: list[str], stop_words: list[str], k: int) -> dict[str, int]:
    unigram_list = [
        word
        for tweet in tweets
        for word in tweet.split()
        if word.lower() not in stop_words and re.match(r"^[a-z#].*", word)
    ]

    top_k_words = Counter(unigram_list)
    return dict(top_k_words) if k == -1 else dict(top_k_words.most_common(k))


def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    # FILL IN CODE
    pass


def pmi(word1, word2, unigram_counter, context_counter):
    # FILL IN CODE
    pass


def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    # FILL IN CODE
    pass


def get_top_k_dimensions(word1_vector, k):
    # FILL IN CODE
    pass


def get_cosine_similarity(word1_vector, word2_vector):
    # FILL IN CODE
    pass


def get_most_similar(word2vec, word, k):
    # FILL IN CODE
    pass


def word_analogy(word2vec, word1, word2, word3):
    # FILL IN CODE
    pass


def cos_sim(A, B):
    # FILL IN CODE
    pass


def get_cos_sim_different_models(word, model1, model2, cos_sim_function):
    # FILL IN CODE
    pass


def get_average_cos_sim(word, neighbors, model, cos_sim_function):
    # FILL IN CODE
    pass


def create_tfidf_matrix(documents, stopwords):
    # FILL IN CODE
    pass


def get_idf_values(documents, stopwords):
    # This part is ungraded, however, to test your code, you'll need to implement this function
    # If you have implemented create_tfidf_matrix, this implementation should be straightforward
    # FILL IN CODE
    pass


def calculate_sparsity(tfidf_matrix):
    # FILL IN CODE
    pass


def extract_salient_words(VT, vocabulary, k):
    # FILL IN CODE
    pass


def get_similar_documents(U, Sigma, VT, doc_index, k):
    # FILL IN CODE
    pass


def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):
    # FILL IN CODE
    pass


if __name__ == "__main__":

    tweets = []
    with open(
        "data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt", "r", encoding="utf-8"
    ) as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open("data/stop_words.txt", "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f.readlines()]

    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)


 # type: ignore