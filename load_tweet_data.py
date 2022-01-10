import pickle
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.keras.utils import to_categorical

EMBEDDING_DIM = 50

def load_tweet_data(tweets_dir, labels_dir, embeddings_dir):
    
    
    # Load tweets, labels
    print("1 -- Loading tweets and labels")
    tweets = tweets_dir
    labels = labels_dir
    for i in range(len(labels)):
        labels[i] /= 4

    # Tokenize the tweets (convert sentence to sequence of words)
    print("2 -- Tokenizing the tweets: converting sentences to sequence of words")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)

    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index

    # Pad sequences to ensure samples are the same size
    print("3 -- Padding sequences to ensure samples are the same size")
    training_data = pad_sequences(sequences)

    print("4 -- Loading pre-trained word embeddings. This may take a few minutes.")

    embeddings_index = {}
    f = open(embeddings_dir,'rb')
    for line in f:
        values = line.split()
        word = values[0].decode('UTF-8')
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print("5 -- Finding word embeddings for words in our tweets.")
    # prepare word embedding matrix
    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    

    return tweets, training_data, labels, word_index, embedding_matrix
