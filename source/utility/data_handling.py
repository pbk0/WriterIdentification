"""
This file is used to store functionality for data handling.
"""
import numpy as np

# global variables
g_train_file = "../data/features/features_train.csv"
g_test_file = "../data/features/features_test.csv"
g_test_sample_entry_file = "../data/solution.csv"


def _read_data(file_name):
    """
    This method takes input the file name and returns back the numpy array of samples and labels.
    :param file_name: the file name to read
    :return: sample and label numpy array
    """
    f = open(file_name)
    # Ignore the header
    f.readline()
    samples_ = []
    label_ = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line[1:]]
        samples_.append(sample)
        label_.append(line[0])
    return np.asarray(samples_), np.asarray(label_)


def _read_solution(file_name):
    """
    Read the solution file and return the label as numpy array.
    :param file_name:
    :return:
    """
    f = open(file_name)
    # Ignore the header
    f.readline()
    label_ = []
    for line in f:
        line = line.strip().split(",")
        label_.append(int(line[1]))
    return np.asarray(label_)


def _read_feature_name(file_name):
    """
    The name of the features used for training
    :param file_name:
    :return:
    """
    f = open(file_name)
    # Ignore the header
    header = f.readline()
    header = header.strip().split(",")
    _feature_name = [x for x in header[1:]]
    return np.asarray(_feature_name)


def load_features_and_labels():
    """
    This method is responsible to fetch the training and test data as numpy array.
    It returns train and test dataset sample and labels.
    It also returns labels of the features used for training.
    :return: _train, _train_label, _test, _test_label, _feature_name
    :rtype: _train, _train_label, _test, _test_label, _feature_name
    """
    global g_train_file, g_test_file, g_test_sample_entry_file

    # get g_train data
    s_, l_ = _read_data(g_train_file)
    l_ = [int(x[:x.index("_")]) for x in l_]
    _train = s_
    _train_label = l_

    # get g_test data
    ss_, ll_ = _read_data(g_test_file)
    _test = ss_

    # get the sample entry
    _test_label = _read_solution(g_test_sample_entry_file)

    # get feature id
    _feature_name = _read_feature_name(g_train_file)

    # print info
    # print("Using features without rbm")

    # return
    return _train, _train_label, _test, _test_label, _feature_name


def write_delimited_file(file_path, data, header=None, delimiter=","):
    """
    Method to save results as a file in the given folder
    :param file_path:
    :param data:
    :param header:
    :param delimiter:
    :return:
    """
    f_out = open(file_path, "w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
    for line in data:
        f_out.write(delimiter.join(line) + "\n")
    f_out.close()


def make_pickle_from_csv():
    """
    Method to save the file as pickle object of numpy arrays.
    Note they are not compatible across platforms so use carefully.
    :return:
    :rtype:
    """
    pass


