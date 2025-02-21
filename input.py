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


def context_word_frequencies(tweets: list[str], stop_words: list[str], context_size: int, frequent_unigrams: list[str]) -> dict[tuple[str, str], int]:
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

    # print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = list(top_k_unigrams(tweets, stop_words, 1000).keys())
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)

    sample_output = context_word_frequencies(tweets, stop_words, 2, unigram_counter)
    print(sample_output.most_common(10))

    
 # type: ignore