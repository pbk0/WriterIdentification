"""
This file is responsible for utility methods to investigate the training of RBMs.
You can only run this code if you have trained RBMs for respective HIDDEN_NEURONS.
"""
import numpy as np
import os

# config
HIDDEN_NEURONS = 4000


def plot_training_error(hidden_neurons):
    """
    Plot the training errors for various epochs
    :param hidden_neurons:
    :type hidden_neurons:
    :return:
    :rtype:
    """
    path = '../data/new_representation/rbm_' + str(hidden_neurons) + '/'
    files = os.listdir(path)
    mean_squared_error = []

    cnt = np.zeros(72, dtype=np.uint8)
    for file in files:
        ff = file.split('_')
        if ff.__len__() > 1:
            cnt[int(ff[1])/50] += 1
        for f in ff:
            if f == 'err':
                err = np.load(path + file)
                epoch = int(ff[1])
                mean_squared_error.append((epoch, float(err)))

    np.asarray(mean_squared_error.sort())
    print(mean_squared_error.__len__())
    print(cnt)


    pass


if __name__ == '__main__':
    plot_training_error(hidden_neurons=HIDDEN_NEURONS)

