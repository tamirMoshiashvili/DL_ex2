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
START = '_START_'
END = '_END_'


def fill_words_and_tags(lines):
    """
    fill the WORDS and TAGS sets from the given lines
    :param lines: list of lines.
    """
    WORDS.update({START, END})
    TAGS.update({START, END})
    for line in lines:
        if line == '':
            continue
        word, tag = line.split()
        WORDS.add(word)
        TAGS.add(tag)


def get_data_set(file_lines):
    """
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


train_pos_filename = 'data/pos/train'
train_pos_lines = read_file(train_pos_filename)
fill_words_and_tags(train_pos_lines)
words_dict = {word: i for i, word in enumerate(WORDS)}
tags_dict = {tag: i for i, tag in enumerate(TAGS)}

dev_pos_filename = 'data/pos/dev'
dev_pos_lines = read_file(dev_pos_filename)

test_pos_filename = 'data/pos/test'

TRAIN = get_data_set(train_pos_lines)
DEV = get_data_set(dev_pos_lines)
TEST = read_file(test_pos_filename)


def to_windows(data_set):
    """

    :param data_set: list of tuples, each tuple is (words, tags) where each of then is a list
    :return: list of tuples, each tuple is (window, tag) where windows is a vector of 5 words,
             each element is represented by its id.
    """
    windows = []
    for words, tags in data_set:
        for i in xrange(2, len(words) - 2):
            curr_window = [words[i-2], words[i-1], words[i], words[i+1], words[i+2]]
            curr_window = [words_dict[word] for word in curr_window]
            windows.append((curr_window, tags_dict[tags[i]]))
    return windows


# if __name__ == '__main__':
#     myfile_name = 'myfile.txt'
#     myfile_lines = read_file(myfile_name)
#
#     data_set = get_data_set(myfile_lines)
#     windows = to_windows(data_set)
#     for window, tag in windows:
#         print 'window: ' + str(window)
#         print 'tag: ' + str(tag)
#         print
