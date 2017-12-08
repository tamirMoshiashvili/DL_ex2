import os
from StringIO import StringIO
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tu_data
from torch.autograd import Variable

import utils


class Net(nn.Module):
    def __init__(self, d_in, d_out, lr, mode, iter_num, d_hid=64, embedding_dim=50, win_size=5):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(d_in, embedding_dim)
        self.linear1 = torch.nn.Linear(win_size * embedding_dim, d_hid)
        self.linear2 = torch.nn.Linear(d_hid, d_out)

        self.emb_dim = embedding_dim
        self.window_size = win_size

        self.mode = mode
        self.O_id = ''
        if MODE == 'NER':
            self.O_id = utils.tags_dict['O']

        # train parameters
        self.batch_size = 1000
        self.lr = lr
        self.iter_num = iter_num

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.window_size * self.emb_dim))
        out = self.linear2(F.tanh(self.linear1(embeds)))
        return out

    def _get_loader_of(self, data):
        """
        NOTE - this function is for train and dev loaders.
        :param data: list of tuples, each tuple is (words, tag) where words is a list of 5 words (all encoded)
        :return: DataLoader object with the given data
        """
        x, y = zip(*data)
        x, y = torch.LongTensor(x), torch.LongTensor(y)
        x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
        data_set = tu_data.TensorDataset(x, y)
        loader = tu_data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        return loader

    def _get_test_loader(self, data):
        """
        NOTE - this function is for test loader.
        :param data: list of words, each is a list of 5 words (all encoded)
        :return: DataLoader object with the given data
        """
        data = torch.LongTensor(data)
        y = torch.ones(data.shape[0], 1)
        y = y.type(torch.LongTensor)
        data, y = data.type(torch.LongTensor), y.type(torch.LongTensor)
        test_set = tu_data.TensorDataset(data, y)
        loader = tu_data.DataLoader(test_set, batch_size=1, shuffle=False)
        return loader

    @staticmethod
    def _write_to_file(fname, text):
        """
        write the content of text in the file.
        :param fname: name of file to write into.
        :param text: StringIO object, contains the content wo write.
        """
        f = open(fname, 'w')
        f.write(text.getvalue())
        f.close()

    def train_on(self, train_data, dev_data, result_filename):
        """
        train the net on the given data and write the results of the dev set in a file (CSV format).
        :param train_data: list of windows, each is list of 5 words.
        :param dev_data: list of windows, each is list of 5 words.
        :param result_filename: name of file to write the result in.
        """
        dev = utils.to_windows(dev_data)
        train_loader = self._get_loader_of(train_data)
        loader_size = len(train_loader)

        criterion = nn.CrossEntropyLoss()  # include the softmax
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        text = StringIO()

        for epoch in range(self.iter_num):
            total_loss = 0.0
            good = bad = 0.0
            curr_t = time()

            for i, data in enumerate(train_loader, 0):
                inputs, tags = data
                inputs, tags = Variable(inputs), Variable(tags)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, tags)
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]

                # extract the predicted label and update good and bad
                _, predicted = torch.max(outputs.data, 1)
                bad += (predicted != tags.data).sum()
                if self.mode == 'NER':
                    zipped = zip(predicted, tags.data)
                    good += sum([pred == real for pred, real in zipped if pred == real and pred != self.O_id])
                else:  # POS
                    good += (predicted == tags.data).sum()

            print str(epoch) + ' - loss: ' + str(total_loss / loader_size) + ', time: ' + str(
                time() - curr_t) + ', accuracy: ' + str(good / (good + bad))

            # check loss and accuracy of dev
            text.write(model.predict_and_check_accuracy(dev, criterion))
        # write to file
        self._write_to_file(result_filename, text)

    def predict_and_check_accuracy(self, data_set, criterion):
        """
        :param data_set: list of windows, each is a list of 5 words.
        :param criterion: loss function.
        :return: string that represent performance, which is 'loss,accuracy'
        """
        good = bad = 0.0
        total_loss = 0.0

        data_loader = self._get_loader_of(data_set)
        loader_size = len(data_loader)
        for i, data in enumerate(data_loader, 0):
            # predict
            inputs, tags = data
            inputs, tags = Variable(inputs), Variable(tags)
            outputs = self(inputs)
            total_loss += criterion(outputs, tags).data[0]
            _, predicted = torch.max(outputs.data, 1)

            # accuracy
            bad += (predicted != tags.data).sum()
            if self.mode == 'NER':
                zipped = zip(predicted, tags.data)
                good += sum([pred == real for pred, real in zipped if pred == real and pred != self.O_id])
            else:  # POS
                good += (predicted == tags.data).sum()
        # loss, acc
        return str(total_loss / loader_size) + ',' + str(good / (good + bad)) + '\n'

    def predict_test(self, test_lines, dest_filename):
        """
        predict on the given test_set, write the results in a file in a format of 'word tag'.
        :param test_lines: test set
        :param dest_filename: name of destination file to write the results into.
        """
        text = StringIO()
        # id_to_word = utils.I2W
        id_to_tag = utils.I2T

        test_loader = self._get_test_loader(utils.test_to_windows(utils.get_test_set(test_lines)))
        for i, data in enumerate(test_loader, 0):
            inputs, _ = data

            # predict
            outputs = self(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            tag = id_to_tag[predicted[0]]

            text.write(tag + '\n')

        words_and_tags = StringIO()
        tags = text.getvalue().splitlines()
        j = 0
        for i in range(len(tags)):
            while test_lines[j] == '':
                words_and_tags.write('\n')
                j += 1
            words_and_tags.write(test_lines[j] + ' ' + tags[i] + '\n')
            j += 1

        # write results to file
        self._write_to_file(dest_filename, words_and_tags)


POS_MODEL_PATH = 'pos_model_file'
NER_MODEL_PATH = 'ner_model_file'

if __name__ == '__main__':
    MODE = 'NER'
    MODEL_PATH = DEV = TRAIN = TEST = ''
    log_filename = ''
    learning_rate = iter_number = 0

    if MODE == 'POS':
        utils.fill_words_and_tags(utils.train_pos_lines)
        MODEL_PATH = POS_MODEL_PATH
        TRAIN = utils.POS_TRAIN
        DEV = utils.POS_DEV
        TEST = utils.POS_TEST
        log_filename = 'pos_dev_log_file'
        learning_rate = 0.001
        iter_number = 10
    else:  # NER
        utils.fill_words_and_tags(utils.train_ner_lines)
        MODEL_PATH = NER_MODEL_PATH
        TRAIN = utils.NER_TRAIN
        DEV = utils.NER_DEV
        TEST = utils.NER_TEST
        log_filename = 'ner_dev_log_file'
        learning_rate = 0.05
        iter_number = 5

    print 'start'
    t = time()

    vocab_size = len(utils.WORDS)
    labels_size = len(utils.TAGS)

    model = Net(vocab_size, labels_size, learning_rate, MODE, iter_number)
    if not os.path.isfile(MODEL_PATH):  # first time
        # train
        model.train_on(utils.to_windows(TRAIN), DEV, log_filename)
        # save the net after train
        torch.save(model.state_dict(), MODEL_PATH)
    else:  # model file exists, was trained before
        model.load_state_dict(torch.load(MODEL_PATH))

    print 'predict test'
    model.predict_test(TEST, 'test1.' + MODE.lower())

    print time() - t
