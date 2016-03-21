"""
This file is responsible for utility methods to analyse the training of RBMs.
You can only run this code if you have trained RBMs for respective HIDDEN_NEURONS.
"""
import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# config
from pandas.io.sas.sas7bdat import _index

HIDDEN_NEURONS = 4000


def plot_training_error(hidden_neurons):
    """
    Plot the training errors for various epochs
    :param hidden_neurons:
    :type hidden_neurons:
    :return:
    :rtype:
    """
    matplotlib.style.use('classic')

    path = '../data/new_representation/rbm_' + str(hidden_neurons) + '/'
    save_path = '../documentation/'
    files = os.listdir(path)
    mean_squared_error = []

    for file in files:
        ff = file.split('_')
        for f in ff:
            if f == 'err':
                err = np.load(path + file)
                epoch = int(ff[1])
                mean_squared_error.append((epoch, float(err)))

    mean_squared_error.sort()
    err_all = np.asarray([v for k, v in mean_squared_error])
    index_all = np.asarray([k for k, v in mean_squared_error])

    df = pd.DataFrame(data=err_all[1:], index=index_all[1:])
    df.columns = ['RBM error']
    ax = df.plot()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Mean squared error of RBM with " + str(hidden_neurons) + " hidden neurons.")
    plt.tight_layout()
    plt.savefig(save_path+"RBM_mse_"+str(hidden_neurons)+".pdf")

    pass


if __name__ == '__main__':
    print(plt.style.available)
    plot_training_error(hidden_neurons=HIDDEN_NEURONS)

