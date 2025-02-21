from sklearn.utils.extmath import randomized_svd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import re
import time


def top_k_unigrams(tweets: list[str], stop_words: list[str], k: int) -> dict[str, int]:
    regex = re.compile(r"^[a-z#].*")
    stop_words = set(stop_words)
    
    unigram_list = [
        word.lower()
        for tweet in tweets
        for word in tweet.split()
        if regex.match(word) and word not in stop_words
    ]

    top_k_words = Counter(unigram_list)
    return top_k_words if k == -1 else dict(top_k_words.most_common(k))


def context_word_frequencies(tweets: list[str], stop_words: list[str], context_size: int, frequent_unigrams) -> dict[tuple[str, str], int]:
    # Convert to set for O(1) lookups
    frequent_unigrams = set(frequent_unigrams) if isinstance(frequent_unigrams, list) else set(frequent_unigrams.keys())
    context_pairs = []
    
    for tweet in tweets:
        # Use numpy array for faster slicing
        tokens = np.array(tweet.lower().split())
        n = len(tokens)
        
        # Create all possible context pairs efficiently
        for i in range(n):
            word1 = tokens[i]
            # Calculate context window boundaries
            start, end = max(0, i - context_size), min(n, i + context_size + 1)
            context = tokens[start:end]
            
            # Filter context words that are in frequent_unigrams
            valid_context = [w for w in context if w in frequent_unigrams and w != word1] # freq_unigrams is a subset of top_k_words
            context_pairs.extend((word1, word2) for word2 in valid_context)
    
    return Counter(context_pairs)

if __name__ == "__main__":

    tweets = []
    with open(
        "data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt", "r", encoding="utf-8"
    ) as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open("data/stop_words.txt", "r", encoding="utf-8") as f:
        stop_words = [line.strip() for line in f.readlines()]

    frequent_unigrams = list(top_k_unigrams(tweets, stop_words, 1000).keys())
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    print(sample_output.most_common(10))
    
 # type: ignore