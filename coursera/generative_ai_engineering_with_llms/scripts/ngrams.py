import os
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams

from utils import preprocess_words


def make_predictions(my_words, freq_grams, vocabulary, normlize=1):
    """
    Generate predictions for the conditional probability of the next word given a sequence.

    Args:
        my_words (list): A list of words in the input sequence.
        freq_grams (dict): A dictionary containing frequency of n-grams.
        normlize (int): A normalization factor for calculating probabilities.
        vocabulary (list): A list of words in the vocabulary.

    Returns:
        list: A list of predicted words along with their probabilities, sorted in descending order.
    """

    vocab_probabilities = {}  # Initialize a dictionary to store predicted word probabilities
    context_size = len(list(freq_grams.keys())[0])  # Determine the context size from n-grams keys

    my_words_ = preprocess_words(my_words)  # Preprocess the input words
    if len(my_words_) < context_size - 1:
        raise ValueError(f"Expected {context_size - 1} context words for this {context_size}-gram model.")

    # Preprocess input words and take only the relevant context words
    my_tokens = my_words_[-(context_size - 1):]

    # Calculate probabilities for each word in the vocabulary given the context
    for next_word in vocabulary:
        temp = my_tokens.copy()
        temp.append(next_word)  # Add the next word to the context

        # Calculate the conditional probability using the frequency information
        if normlize != 0:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] / normlize
        else:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] 
    # Sort the predicted words based on their probabilities in descending order
    vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)

    return vocab_probabilities  # Return the sorted list of predicted words and their probabilities


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), '..\\Labs\\Lab3\\song.txt'), 'r') as file:
        song = file.read()

    unigrams = preprocess_words(song)
    print(unigrams[0:10])

    ### Unigram model
    # Create a frequency distribution of words
    fdist = FreqDist(unigrams)
    print(fdist.most_common(10))

    # Create a vocabulary set from the unigrams
    vocabulary = set(unigrams)

    ### Bigram model
    bigrams_ = bigrams(unigrams)
    freq_bigrams  = FreqDist(bigrams_)
    print(freq_bigrams.most_common(10))

    # Example usage of make_predictions
    # Predict the next word given a sequence of words
    my_words = 'are'
    normlize = fdist.get(my_words, 1)  # Get the frequency of the word or default to 1 if not found
    vocab_probabilities = make_predictions(
        my_words=my_words,
        freq_grams=freq_bigrams,
        vocabulary=vocabulary,
        normlize=normlize)
    print(vocab_probabilities[0:10])  # Print the top 10 predicted words and their probabilities

    # Generate a song using the bigram model
    my_song = "i"
    my_word = my_song
    for i in range(100):
        my_word = make_predictions(
            my_words=my_word,
            freq_grams=freq_bigrams,
            vocabulary=vocabulary)[0][0]
        my_song += " " + my_word
    print(my_song)

    ### Trigram model
    trigrams_ = trigrams(unigrams)
    freq_trigrams  = FreqDist(trigrams_)
    print(freq_bigrams.most_common(10))

    # Generate a song using the trigram model
    my_song = "i just"
    my_word = my_song
    for i in range(100):
        my_word = make_predictions(
            my_words=my_word,
            freq_grams=freq_trigrams,
            vocabulary=vocabulary)[0][0]
        my_song += " " + my_word
        my_word = f'{my_song.split()[-2]} {my_song.split()[-1]}'  # Use the last two words as context
    print(my_song)
    