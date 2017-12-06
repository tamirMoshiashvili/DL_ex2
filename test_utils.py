from StringIO import StringIO
from utils import read_file


def convert_to_test_file(src_name, dest_name):
    f = open(src_name, 'r')
    file_lines = f.read().splitlines()
    f.close()

    text = StringIO()

    for line in file_lines:
        if line != '':
            text.write(line.split()[0])
        text.write('\n')

    f = open(dest_name, 'w')
    f.write(text.getvalue())
    f.close()


def comp_files(src_filename, pred_filename):
    src = read_file(src_filename)
    src = [line for line in src if line != '']
    pred = read_file(pred_filename)

    if len(src) != len(pred):
        print 'length of files are different'
        return

    good = bad = 0.0
    for i in range(len(src)):
        src_line = src[i]
        if src_line == '':
            continue
        pred_line = pred[i]

        w1, t1 = src_line.split()
        w2, t2 = pred_line.split()

        if w2 != '_UNK_' and w1 != w2:
            print 'words are different'
            return

        if t1 == t2:
            good += 1
        else:
            bad += 1

    print 'acc: ' + str(good / (good + bad))


if __name__ == '__main__':
    comp_files('data/pos/train', 'pos_train.pred')
    comp_files('data/pos/dev', 'pos_dev.pred')
