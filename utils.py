TAGS = set()
WORDS = set()
I2W = dict()
I2T = dict()
words_dict = dict()
tags_dict = dict()


def read_file(filename):
    """
    :param filename name of file to read
    :return list of lines.
    """
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines


# unique tokens
UNK = '_UNK_'
START = '_START_'
END = '_END_'


def fill_words_and_tags(lines):
    """
    fill the WORDS and TAGS sets from the given lines
    :param lines: list of lines.
    """
    # include the unique tokens
    WORDS.update({START, END, UNK})
    TAGS.update({START, END})

    for line in lines:
        if line == '':  # ignore empty lines
            continue

        word, tag = line.split()
        WORDS.add(word)
        TAGS.add(tag)
    # dicts for encode and decode the words and tags
    I2W.update({i: word for i, word in enumerate(WORDS)})
    words_dict.update({word: i for i, word in enumerate(WORDS)})
    I2T.update({i: tag for i, tag in enumerate(TAGS)})
    tags_dict.update({tag: i for i, tag in enumerate(TAGS)})


def get_data_set(file_lines):
    """
    NOTE -  file_lines must include the tag along with the word,
            this function is for train and dev data.
    :param file_lines: list of lines, each line is 'word tag' or ''
    :return: list of tuples, each tuple is sentence and its tags.
    """
    sentence = []
    sentence_tags = []
    lines = []

    file_lines.append('')
    for line in file_lines:
        if line == '':
            # start tags
            sentence.insert(0, START)
            sentence.insert(0, START)
            sentence_tags.insert(0, START)
            sentence_tags.insert(0, START)

            # end tags
            sentence.extend([END, END])
            sentence_tags.extend([END, END])

            # insert the current tuple (sentence, tags) and clear the lists
            lines.append((sentence, sentence_tags))
            sentence = []
            sentence_tags = []
        else:
            # add the word and tag
            word, tag = line.split()
            sentence.append(word)
            sentence_tags.append(tag)
    return lines


def get_test_set(file_lines):
    """
    NOTE -  each line in file_lines is a single word,
            this function is for test data.
    :param file_lines: list of lines, each line is 'word' or ''
    :return: list of sentences.
    """
    sentence = []
    lines = []

    file_lines.append('')
    for line in file_lines:
        if line == '':
            # start tags
            sentence.insert(0, START)
            sentence.insert(0, START)

            # end tags
            sentence.extend([END, END])

            # insert the current tuple (sentence, tags) and clear the lists
            lines.append(sentence)
            sentence = []
        else:
            sentence.append(line)
    return lines


# TRAIN
train_pos_filename = 'data/pos/train'
train_pos_lines = read_file(train_pos_filename)
train_ner_filename = 'data/ner/train'
train_ner_lines = read_file(train_ner_filename)

# DEV
dev_pos_filename = 'data/pos/dev'
dev_pos_lines = read_file(dev_pos_filename)
dev_ner_filename = 'data/ner/dev'
dev_ner_lines = read_file(dev_ner_filename)

# TEST
test_pos_filename = 'data/pos/test'
test_pos_lines = read_file(test_pos_filename)
test_ner_filename = 'data/ner/test'
test_ner_lines = read_file(test_ner_filename)

# create the sets
POS_TRAIN = get_data_set(train_pos_lines)
POS_DEV = get_data_set(dev_pos_lines)
POS_TEST = get_test_set(test_pos_lines)

NER_TRAIN = get_data_set(train_ner_lines)
NER_DEV = get_data_set(dev_ner_lines)
NER_TEST = get_test_set(test_ner_lines)


def to_windows(data_set):
    """
    NOTE - data_set must include the tags along side the words,
            this function is for train and dev data sets.
    :param data_set: list of tuples, each tuple is (words, tags) where each of them is a list
    :return: list of tuples, each tuple is (window, tag) where window is a vector of 5 words,
             each element is represented by its id i.e. number.
    """
    windows = []
    for words, tags in data_set:
        for i in xrange(2, len(words) - 2):
            # create window of words
            curr_window = [words[i - 2], words[i - 1], words[i], words[i + 1], words[i + 2]]

            # replace the unknown words in UNK
            for j, word in enumerate(curr_window):
                if word not in WORDS:
                    curr_window[j] = UNK

            # encode
            curr_window = [words_dict[word] for word in curr_window]
            windows.append((curr_window, tags_dict[tags[i]]))
    return windows


def test_to_windows(test_set):
    """
    NOTE - test_set must be *only* words,
            this function is for test set.
    :param test_set: list of words where each of them is a list
    :return: list of windows, each window is a vector of 5 words,
             each element is represented by its id i.e. number.
    """
    windows = []
    for words in test_set:
        for i in xrange(2, len(words) - 2):
            # create window of words
            curr_window = [words[i - 2], words[i - 1], words[i], words[i + 1], words[i + 2]]

            # replace the unknown words in UNK
            for j, word in enumerate(curr_window):
                if word not in WORDS:
                    curr_window[j] = UNK

            # encode
            curr_window = [words_dict[word] for word in curr_window]
            windows.append(curr_window)
    return windows
