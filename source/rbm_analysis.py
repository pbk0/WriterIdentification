"""
This file is responsible for utility methods to analyse the training of RBMs.
You can only run this code if you have trained RBMs for respective HIDDEN_NEURONS.
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
"""
import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame

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


def build_hidden_activities(hidden_neurons, select_epochs):
    """

    This routine plots hidden activities based on test data.
    This helps us to visualize how the training of RBM progressed as error reduces.

    :param hidden_neurons:
    :type hidden_neurons:
    :param select_epochs: select epocs to plot
    :type select_epochs: list
    :return:
    :rtype:
    """
    from source.utility.data_handling import load_features_and_labels
    from collections import namedtuple
    import csv

    directory = "../data/new_representation/rbm_" + str(hidden_neurons) + "/_"

    # load features
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()
    _train = np.transpose(np.asarray(g_train))
    _test = np.transpose(np.asarray(g_test))
    _random = np.random.rand(_train.shape[0], _train.shape[1])

    # hidden activity
    test_hidden_activities = []
    test_hidden_activities_bin = []
    train_hidden_activities = []
    train_hidden_activities_bin = []
    random_hidden_activities = []
    random_hidden_activities_bin = []
    activities_epochs = []
    activities_mse = []

    # get the epochs
    for epoch in select_epochs:

        print('Working on epoch ' + str(epoch) + '\n')

        # load model
        w_vh = np.load(directory + str(epoch) + '_numpy_w_vh_.npy')
        w_v = np.load(directory + str(epoch) + '_numpy_w_v_.npy')
        w_h = np.load(directory + str(epoch) + '_numpy_w_h_.npy')
        err = np.load(directory + str(epoch) + '_numpy_err_.npy')

        #
        activities_epochs.append(epoch)
        activities_mse.append(err)

        #######################################################
        # positive phase
        # g_train_h = 1. / (1 + np.exp(-(np.dot(w_vh.T, _train) + w_h)))
        h_test = 1. / (1 + np.exp(-(np.dot(w_vh.T, _test) + w_h)))

        # sample hidden activity
        h_test1 = 1. * (h_test > np.random.rand(hidden_neurons, h_test.shape[1]))
        h_test1 = np.transpose(h_test1)
        h_test1 = np.asarray(sum(h_test1))

        # add to list bin
        test_hidden_activities_bin.append(h_test1)

        # add to list
        h_test2 = np.transpose(h_test)
        h_test2 = np.asarray(sum(h_test2))
        test_hidden_activities.append(h_test2)

        #######################################################
        # positive phase
        # g_train_h = 1. / (1 + np.exp(-(np.dot(w_vh.T, _train) + w_h)))
        h_test = 1. / (1 + np.exp(-(np.dot(w_vh.T, _train) + w_h)))

        # sample hidden activity
        h_test1 = 1. * (h_test > np.random.rand(hidden_neurons, h_test.shape[1]))
        h_test1 = np.transpose(h_test1)
        h_test1 = np.asarray(sum(h_test1))

        # add to list bin
        train_hidden_activities_bin.append(h_test1)

        # add to list
        h_test2 = np.transpose(h_test)
        h_test2 = np.asarray(sum(h_test2))
        train_hidden_activities.append(h_test2)

        #######################################################
        # positive phase
        # g_train_h = 1. / (1 + np.exp(-(np.dot(w_vh.T, _train) + w_h)))
        h_test = 1. / (1 + np.exp(-(np.dot(w_vh.T, _random) + w_h)))

        # sample hidden activity
        h_test1 = 1. * (h_test > np.random.rand(hidden_neurons, h_test.shape[1]))
        h_test1 = np.transpose(h_test1)
        h_test1 = np.asarray(sum(h_test1))

        # add to list bin
        random_hidden_activities_bin.append(h_test1)

        # add to list
        h_test2 = np.transpose(h_test)
        h_test2 = np.asarray(sum(h_test2))
        random_hidden_activities.append(h_test2)

    # store to file
    test_hidden_activities = np.asarray(test_hidden_activities)
    test_hidden_activities_bin = np.asarray(test_hidden_activities_bin)
    train_hidden_activities = np.asarray(train_hidden_activities)
    train_hidden_activities_bin = np.asarray(train_hidden_activities_bin)
    random_hidden_activities = np.asarray(train_hidden_activities)
    random_hidden_activities_bin = np.asarray(train_hidden_activities_bin)
    activities_epochs = np.asarray(activities_epochs)
    activities_mse = np.asarray(activities_mse)
    np.savez_compressed(
        file=directory + 'hidden_activities',
        test_hidden_activities=test_hidden_activities,
        test_hidden_activities_bin=test_hidden_activities_bin,
        train_hidden_activities=train_hidden_activities,
        train_hidden_activities_bin=train_hidden_activities_bin,
        random_hidden_activities=random_hidden_activities,
        random_hidden_activities_bin=random_hidden_activities_bin,
        activities_epochs=activities_epochs,
        activities_mse=activities_mse
    )


def plot_hidden_activities(hidden_neurons, epoch_slice, hidden_neuron_slice):
    """

    This routine plots hidden activities based on test data.
    This helps us to visualize how the training of RBM progressed as error reduces.

    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html

    :param hidden_neuron_slice: decide which hidden neuron activities to plot
    :type hidden_neuron_slice:
    :param epoch_slice: decide which epochs to consider
    :type epoch_slice:
    :param hidden_neurons:
    :type hidden_neurons: int
    :return:
    :rtype:
    """
    import seaborn as sns
    sns.set()

    directory = "../data/new_representation/rbm_" + str(hidden_neurons) + "/_"
    save_path = '../documentation/'

    npzfile = np.load(directory + 'hidden_activities.npz')
    test_hidden_activities = np.asarray(npzfile['test_hidden_activities'][np.ix_(epoch_slice, hidden_neuron_slice)])
    test_hidden_activities_bin = np.asarray(npzfile['test_hidden_activities_bin'][np.ix_(epoch_slice, hidden_neuron_slice)])
    train_hidden_activities = np.asarray(npzfile['train_hidden_activities'][np.ix_(epoch_slice, hidden_neuron_slice)])
    train_hidden_activities_bin = np.asarray(npzfile['train_hidden_activities_bin'][np.ix_(epoch_slice, hidden_neuron_slice)])
    random_hidden_activities = np.asarray(npzfile['random_hidden_activities'][np.ix_(epoch_slice, hidden_neuron_slice)])
    random_hidden_activities_bin = np.asarray(npzfile['random_hidden_activities_bin'][np.ix_(epoch_slice, hidden_neuron_slice)])
    activities_epochs = np.asarray(npzfile['activities_epochs'][epoch_slice])
    activities_mse = np.asarray(npzfile['activities_mse'][epoch_slice])

    # test_hidden_activities
    data_arr=np.transpose(random_hidden_activities_bin)
    print('Doing ... ')
    df = DataFrame(data=data_arr)
    df.plot(subplots=True, figsize=(8, 8))
    plt.legend(loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path+"RBM_hidden_activities_"+str(hidden_neurons)+".pdf")
    plt.clf()
    np.savetxt(save_path+"RBM_hidden_activities_"+str(hidden_neurons)+".csv", data_arr, delimiter=',')


if __name__ == '__main__':

    # training error analysis
    if False:
        plot_training_error(hidden_neurons=HIDDEN_NEURONS)

    # hidden activity analysis
    #select_epochs_to_analyze = [0, 1, 2, 3, 4, 5, 6, 1000, 3000, 3550]
    select_epochs_to_analyze = [0,1,2,4,8,16,32,100,150]
    if False:
        build_hidden_activities(hidden_neurons=HIDDEN_NEURONS, select_epochs=select_epochs_to_analyze)
    if True:
        plot_hidden_activities(
            hidden_neurons=HIDDEN_NEURONS,
            epoch_slice=np.arange(select_epochs_to_analyze.__len__()),
            hidden_neuron_slice=np.arange(0, 100)
        )

