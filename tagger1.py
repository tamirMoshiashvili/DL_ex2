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
        self.lr = 2

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
        data, y = torch.LongTensor(data), torch.LongTensor()
        data, y = data.type(torch.LongTensor), y.type(torch.LongTensor)
        test_set = tu_data.TensorDataset(data, y)
        loader = tu_data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return loader

    def train_on(self, train_data):
        dev = utils.to_windows(utils.DEV)
        train_loader = self._get_loader_of(train_data)
        loader_size = len(train_loader)

        criterion = nn.CrossEntropyLoss()  # include the softmax
        optimizer = optim.SGD(self.parameters(), lr=self.lr)

        for epoch in range(10):
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
            net.predict_and_check_accuracy(dev, 'DEV')

    def predict_and_check_accuracy(self, data_set, name):
        good = bad = 0.0

        data_loader = self._get_loader_of(data_set)
        for i, data in enumerate(data_loader, 0):
            # predict
            inputs, tags = data
            outputs = self(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)

            # accuracy
            bad += (predicted != tags).sum()
            good += (predicted == tags).sum()
        print '\t' + name + ' accuracy: ' + str(good / (good + bad))

    def predict_test(self, test_set):
        text = StringIO()
        id_to_word = utils.I2W
        id_to_tag = utils.I2T

        test_loader = self._get_test_loader(test_set)
        for i, inputs in enumerate(test_loader, 0):
            # predict
            outputs = self(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            word = inputs[2]

            text.write(id_to_word[word] + ' ' + id_to_tag[predicted] + '\n')


if __name__ == '__main__':
    print 'start'
    t = time()

    train_data = utils.TRAIN
    vocab_size = len(utils.WORDS)
    lables_size = len(utils.TAGS)
    window_size = 5
    hidden_dim = 40
    net = Net(vocab_size, window_size, hidden_dim, lables_size)
    net.train_on(utils.to_windows(train_data))

    # net.predict_and_check_accuracy(utils.to_windows(utils.TEST), 'TEST')

    print time() - t
