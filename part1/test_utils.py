from StringIO import StringIO
from utils import read_file


def convert_to_test_file(src_name, dest_name):
    """
    create a file that the net can treat as a blind test.
    :param src_name: name of source file, which is lines of 'word tag'.
    :param dest_name: name of the file to create, will eventually contain lines where each line is 'word'
    """
    # read the source file
    f = open(src_name, 'r')
    file_lines = f.read().splitlines()
    f.close()

    # save the needed data in text
    text = StringIO()
    for line in file_lines:
        if line != '':
            text.write(line.split()[0])
        text.write('\n')

    # create file and write output to it
    f = open(dest_name, 'w')
    f.write(text.getvalue())
    f.close()


def comp_files(src_filename, pred_filename):
    """
    compare 2 files and check the net accuracy.
    :param src_filename: name of source file, which is the original file, contains the real tag for each word.
    :param pred_filename: name of prediction file, contains the new's prediction for each word.
    """
    src = read_file(src_filename)
    src = [line for line in src if line != '']  # filter empty lines ('')
    pred = read_file(pred_filename)

    if len(src) != len(pred):
        print 'Error: length of files are different'
        return

    good = bad = 0.0
    for i in range(len(src)):
        # extract lines of both files
        src_line = src[i]
        pred_line = pred[i]

        # extract words and tags from each line
        w1, t1 = src_line.split()
        w2, t2 = pred_line.split()

        if w2 != '_UNK_' and w1 != w2:
            print 'Error: words are different'
            return

        # update
        if t1 == t2:
            good += 1
        else:
            bad += 1

    print 'acc: ' + str(good / (good + bad))


if __name__ == '__main__':
    # comp_files('data/pos/train', 'pos_train.pred')
    comp_files('data/pos/dev', 'pos_dev.pred')
