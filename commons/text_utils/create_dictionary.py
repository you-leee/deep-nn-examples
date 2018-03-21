import collections
from itertools import chain


def create_dictionary(tokenized_docs, min_threshold, max_threshold, max_counts_n, min_counts_n):
    # Gets a list of lists, where each inner list is a tokenized version (list of words filtered) of a text
    # Returns the dictionary containing the set of words from all the texts and also the most and least popular words with their count

    all_words = list(chain(*tokenized_docs))
    word_counts = collections.Counter(all_words)
    filtered_word_counts = collections.Counter({x: word_counts[x] for x in word_counts if
                                                (word_counts[x] >= min_threshold and word_counts[x] <= max_threshold)})
    filtered_words = list(filtered_word_counts.keys())
    dictionary = set(filtered_words)

    most_common_words = filtered_word_counts.most_common()
    return dictionary, most_common_words[:max_counts_n], most_common_words[:-min_counts_n - 1:-1]
