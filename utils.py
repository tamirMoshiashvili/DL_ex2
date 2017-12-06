TAGS = set()
WORDS = set()


def read_file(filename):
    """
    get filename.
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
    WORDS.update({START, END, UNK})
    TAGS.update({START, END})
    for line in lines:
        if line == '':
            continue
        word, tag = line.split()
        WORDS.add(word)
        TAGS.add(tag)


def get_data_set(file_lines, contains_tags=True):
    """
    :param file_lines: list of lines, each line is 'word tag' or 'word' (can be also '')
    :return: list of tuples, each tuple is sentence and its tags,
             or list of sentences.
    """
    sentence = []
    if contains_tags:
        sentence_tags = []
    lines = []

    file_lines.append('')
    for line in file_lines:
        if line == '':
            # start tags
            sentence.insert(0, START)
            sentence.insert(0, START)
            if contains_tags:
                sentence_tags.insert(0, START)
                sentence_tags.insert(0, START)
                sentence_tags.extend([END, END])

            # end tags
            sentence.extend([END, END])

            # insert the current tuple (sentence, tags) and clear the lists
            sentence = []
            if contains_tags:
                lines.append((sentence, sentence_tags))
                sentence_tags = []
            else:
                lines.append(sentence)
        else:
            # add the word and tag
            if contains_tags:
                word, tag = line.split()
                sentence.append(word)
                sentence_tags.append(tag)
            else:
                sentence.append(line)
    return lines


train_pos_filename = 'data/pos/train'
train_pos_lines = read_file(train_pos_filename)
fill_words_and_tags(train_pos_lines)

I2W = enumerate(WORDS)
words_dict = {word: i for i, word in I2W}

I2T = enumerate(TAGS)
tags_dict = {tag: i for i, tag in I2T}

dev_pos_filename = 'data/pos/dev'
dev_pos_lines = read_file(dev_pos_filename)

test_pos_filename = 'data/pos/test'
test_pos_lines = read_file(test_pos_filename)

TRAIN = get_data_set(train_pos_lines)
DEV = get_data_set(dev_pos_lines)
TEST = get_data_set(test_pos_lines, contains_tags=False)


def to_windows(data_set):
    """
    :param data_set: list of tuples, each tuple is (words, tags) where each of them is a list
    :return: list of tuples, each tuple is (window, tag) where windows is a vector of 5 words,
             each element is represented by its id.
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


def test_to_window(test_set):
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
