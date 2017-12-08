from time import time

import numpy as np
import sys


def read_vocab_file(filename):
    """
    :param filename: name of vocabulary file.
    :return: list of strings, where each is a word.
    """
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


vocab = read_vocab_file('vocab.txt')
vecs = np.loadtxt('wordVectors.txt')
vocab_size = len(vocab)


def dist(u, v):
    """
    calculate the distance between two vectors, according to the cosine distance.
    :param u: vector.
    :param v: vector.
    :return: number representing the distance.
    """
    numerator = np.dot(u, v)
    denominator = np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))
    return numerator / denominator


def most_similar(word, k):
    """
    get the most k similar words to the given word out od the vocabulary.
    :param word: word to find similarities to.
    :param k: number of similar words to find.
    :return: list of tuples, each tuple is a (word, distance).
    """
    # convert the word to a vector
    word_index = vocab.index(word)
    word = vecs[word_index]

    dist_vec = np.empty(vocab_size)

    for i, v in enumerate(vecs):
        if i == word_index:  # skip the calculation of distance between the word to itself
            continue
        dist_vec[i] = dist(word, v)

    # extract most similar words according to the calculated results
    ind = np.argpartition(dist_vec, -k)[-k:]
    return zip([vocab[i] for i in ind], [dist_vec[i] for i in ind])


if __name__ == '__main__':
    t = time()
    print 'start'

    words = ['dog', 'england', 'john', 'explode', 'office']
    num = int(sys.argv[1])  # get number of similar words from the passed arguments
    for w in words:
        print w + ': ' + str(most_similar(w, num))

    print time() - t
