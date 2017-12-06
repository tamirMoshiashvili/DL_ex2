from time import time

from StringIO import StringIO

import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as tu_data


class Net(nn.Module):
    def __init__(self, d_in, win_size, d_hid, d_out, embedding_dim=50):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(d_in, embedding_dim)
        self.linear1 = torch.nn.Linear(win_size * embedding_dim, d_hid)
        self.linear2 = torch.nn.Linear(d_hid, d_out)

        self.emb_dim = embedding_dim
        self.window_size = win_size

        self.batch_size = 1000
        self.lr = 0.001
        self.iter_num = 10

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.window_size * self.emb_dim))
        out = self.linear2(F.tanh(self.linear1(embeds)))
        return out

    def _get_loader_of(self, data):
        """
        :param data: list of tuples, each tuple is (words, tag) where words is a list of 5 words (all ecoded)
        :return: DataLoader object with the given data
        """
        x, y = zip(*data)
        x, y = torch.LongTensor(x), torch.LongTensor(y)
        x, y = x.type(torch.LongTensor), y.type(torch.LongTensor)
        data_set = tu_data.TensorDataset(x, y)
        loader = tu_data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        return loader

    def _get_test_loader(self, data):
        data = torch.LongTensor(data)
        y = torch.ones(data.shape[0], 1)
        y = y.type(torch.LongTensor)
        data, y = data.type(torch.LongTensor), y.type(torch.LongTensor)
        test_set = tu_data.TensorDataset(data, y)
        loader = tu_data.DataLoader(test_set, batch_size=1, shuffle=False)
        return loader

    def train_on(self, train_data):
        dev = utils.to_windows(utils.DEV)
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

                _, predicted = torch.max(outputs.data, 1)
                bad += (predicted != tags.data).sum()
                good += (predicted == tags.data).sum()

            print str(epoch) + ' - loss: ' + str(total_loss / loader_size) + ', time: ' + str(
                time() - curr_t) + ', accuracy: ' + str(good / (good + bad))

            # check loss and acc of dev
            text.write(model.predict_and_check_accuracy(dev, criterion))

        # write to file
        self._write_to_file('dev_log_file', text)

    @staticmethod
    def _write_to_file(fname, text):
        f = open(fname, 'w')
        f.write(text.getvalue())
        f.close()

    def predict_and_check_accuracy(self, data_set, criterion):
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
            good += (predicted == tags.data).sum()
        # loss, acc
        return str(total_loss / loader_size) + ',' + str(good / (good + bad)) + '\n'

    def predict_test(self, test_set, dest_filename):
        text = StringIO()
        id_to_word = utils.I2W
        id_to_tag = utils.I2T

        test_loader = self._get_test_loader(test_set)
        for i, data in enumerate(test_loader, 0):
            inputs, _ = data
            # predict
            outputs = self(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            word = id_to_word[inputs[0][2]]
            tag = id_to_tag[predicted[0]]

            text.write(word + ' ' + tag + '\n')

        # write results to file
        self._write_to_file(dest_filename, text)


MODEL_PATH = 'model_file'

if __name__ == '__main__':
    print 'start'
    t = time()

    train_data = utils.TRAIN
    vocab_size = len(utils.WORDS)
    lables_size = len(utils.TAGS)
    window_size = 5
    hidden_dim = 64
    model = Net(vocab_size, window_size, hidden_dim, lables_size)
    model.train_on(utils.to_windows(train_data))

    # save the net after train
    torch.save(model.state_dict(), MODEL_PATH)

    # to load the model do the following
    # model = Net(params)
    # model.load_state_dict(torch.load(MODEL_PATH))

    print 'predict train'
    model.predict_test(utils.test_to_windows(utils.get_test_set(utils.read_file('test_data/train_pos_test'))),
                       'pos_train.pred')

    print 'predict dev'
    model.predict_test(utils.test_to_windows(utils.get_test_set(utils.read_file('test_data/dev_pos_test'))),
                       'pos_dev.pred')

    print time() - t
