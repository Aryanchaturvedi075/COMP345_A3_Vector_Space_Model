from sklearn.utils.extmath import randomized_svd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
import gensim
import re

import pickle
from time import perf_counter

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


def context_word_frequencies(tweets: list[str], stop_words: list[str], context_size: int, frequent_unigrams) -> dict[(str, str), int]:
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
            valid_context = [w for w in context if w in frequent_unigrams and w != word1] # frequent_unigrams is a subset of top_k_words
            context_pairs.extend((word1, word2) for word2 in valid_context)
    
    return Counter(context_pairs)


def pmi(word1: str, word2: str, unigram_counter: dict[str, int], context_counter: dict[(str, str), int]) -> float:
    total_unigrams = float(sum(unigram_counter.values()))
    total_bigrams = float(sum(context_counter.values()))
    
    # Get the counts (with pseudo-count = 1 if not observed)
    count_w1 = float(unigram_counter.get(word1, 1))
    count_w2 = float(unigram_counter.get(word2, 1))
    count_w1_w2 = float(context_counter.get((word1, word2), 1))
    
    p_w1 = count_w1 / total_unigrams
    p_w2 = count_w2 / total_unigrams
    p_w1_w2 = count_w1_w2 / total_bigrams
    
    return np.log2(p_w1_w2 / (p_w1 * p_w2))


def build_word_vector(word1: str, frequent_unigrams, unigram_counter: dict[str, int], context_counter: dict[(str, str), int]) -> dict[str, float]:
    frequent_unigrams = set(frequent_unigrams) if isinstance(frequent_unigrams, list) else set(frequent_unigrams.keys())
    context_set = set(context_counter.keys())
    word_vector = {}

    for word2 in frequent_unigrams:
        word_vector[word2] = float(0) if (word1, word2) not in context_set else pmi(word1, word2, unigram_counter, context_counter)
    
    return word_vector


def get_top_k_dimensions(word1_vector, k):
    sorted_items = sorted(word1_vector.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:k])


def get_cosine_similarity(word1_vector: dict[str, float], word2_vector: dict[str, float]) -> float:
    # Convert dictionaries to numpy arrays
    vec1 = np.array([word1_vector.get(word) for word in word1_vector.keys()])
    vec2 = np.array([word2_vector.get(word) for word in word2_vector.keys()])
    
    # Use numpy's optimized operations
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def get_most_similar(word2vec: gensim.models.keyedvectors.KeyedVectors, word : str, k : int) -> list[(str, float)]:

    print("reached here")
    # Check if the word exists in the model's vocabulary
    if word not in word2vec.key_to_index:
        return []
    
    try:
        # Use gensim's most_similar method to find k most similar words
        # Returns list of tuples (word, similarity)
        similar_words = word2vec.most_similar(word, topn=k)
        return similar_words
        
    except KeyError:
        # Handle any potential KeyError that might occur
        return []


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


def load_or_compute_variables():
    tic = perf_counter()
    try:
        # Load all objects from the file
        with open('twitter_analysis_data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        # Access individual variables
        tweets = loaded_data['tweets']
        stop_words = loaded_data['stop_words']
        frequent_unigrams = loaded_data['frequent_unigrams']
        unigram_counter = loaded_data['unigram_counter']
        context_counter = loaded_data['context_counter']

        print(f"Loaded {len(tweets)} tweets and {len(frequent_unigrams)} frequent unigrams")
        toc = perf_counter()
        print(f"Time taken: {toc - tic}s")
        return tweets, stop_words, frequent_unigrams, unigram_counter, context_counter

    except Exception as e:
        print(f"Cache invalid, recomputing: {str(e)}")

    # Compute fresh values if cache is invalid/missing
    print("Computing fresh values...")
    tweets, stop_words = [], []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt', "r", encoding='utf-8') as f:
        tweets = [line.strip() for line in f.readlines()]
    with open('data/stop_words.txt', "r", encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    print(f"computed {len(tweets)} tweets and {len(stop_words)} stop words in {perf_counter() - tic:.2f} seconds")
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    print(f"computed {len(frequent_unigrams)} frequent_unigrams in {perf_counter() - tic:.2f} seconds")
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    print(f"computed {len(unigram_counter)} unigram_counter in {perf_counter() - tic:.2f} seconds")
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)
    print(f"computed {len(context_counter)} context_counter in {perf_counter() - tic:.2f} seconds")

    # EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    # print(f"loaded word2vec model in {perf_counter() - tic:.2f} seconds")

    # Save to cache with metadata
    objects_to_save = {
        'tweets': tweets,
        'stop_words': stop_words,
        'frequent_unigrams': frequent_unigrams,
        'unigram_counter': unigram_counter,
        'context_counter': context_counter
    }

    # Save all objects to a single file
    with open('twitter_analysis_data.pkl', 'wb') as f:
        pickle.dump(objects_to_save, f)

    print(f"Successfully saved {len(objects_to_save)} objects to twitter_analysis_data.pkl")
    toc = perf_counter()
    print(f"Time taken: {toc - tic}s")
    return tweets, stop_words, frequent_unigrams, unigram_counter, context_counter


# global variables
tweets, stop_words, frequent_unigrams, unigram_counter, context_counter = load_or_compute_variables()


if __name__ == '__main__':

    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    tic = perf_counter()
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    toc = perf_counter()
    print(f"Time taken to load Word2Vec model: {toc - tic:.2f} seconds")

    print("Does it even get here?")
    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    """Word2Vec for Meaning Change"""

    # Comparing 40-60 year olds in the 1910s and 40-60 year olds in the 2000s
    model_t1 = Word2Vec.load('data/1910s_50yos.model')
    model_t2 = Word2Vec.load('data/2000s_50yos.model')

    # Cosine similarity function for vector inputs
    vector_1 = np.array([1,2,3,4])
    vector_2 = np.array([3,5,4,2])
    cos_similarity = cos_sim(vector_1, vector_2)
    print(cos_similarity)
    # 0.8198915917499229

    # Similarity between embeddings of the same word from different times
    major_cos_similarity = get_cos_sim_different_models("major", model_t1, model_t2, cos_sim)
    print(major_cos_similarity)
    # 0.19302374124526978

    # Average cosine similarity to neighborhood of words
    neighbors_old = ['brigadier', 'colonel', 'lieutenant', 'brevet', 'outrank']
    neighbors_new = ['significant', 'key', 'big', 'biggest', 'huge']
    print(get_average_cos_sim("major", neighbors_old, model_t1, cos_sim))
    # 0.6957747220993042
    print(get_average_cos_sim("major", neighbors_new, model_t1, cos_sim))
    # 0.27042335271835327
    print(get_average_cos_sim("major", neighbors_old, model_t2, cos_sim))
    # 0.2626224756240845
    print(get_average_cos_sim("major", neighbors_new, model_t2, cos_sim))
    # 0.6279034614562988

    ### The takeaway -- When comparing word embeddings from 40-60 year olds in the 1910s and 2000s,
    ###                 (i) cosine similarity to the neighborhood of words related to military ranks goes down;
    ###                 (ii) cosine similarity to the neighborhood of words related to significance goes up.


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # type: ignore # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
