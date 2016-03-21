"""
Code to train generative RBM model

You just need to run this file to train the model
Learned model is saved as weights in data/rbm_learned_weights/ directory
We dump learned weights after every 100 epochs

Later this learned weights can be used to generate hidden activities

Feature addition: Now you can resume your training from any iteration and continue learning :)
"""

from __future__ import division
from source.utility.data_handling import load_features_and_labels
import time
import numpy as np


def train_rbm(num_hid, start_epoch, end_epoch):
    """
    Train the RBM and dump learned weights as numpy errors in data folder.
    :param num_hid: number of hidden neuron to train model for.
    :type num_hid: int
    :param start_epoch: the starting epoch. If you have previous iteration results saved you can continue from here :)
    :type start_epoch: int
    :param end_epoch: The last epoch you want to have
    :type end_epoch: int
    :return:
    :rtype:
    """
    # load data
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()
    # util.load('mnist.dat', globals())
    # dat = dat/255.
    # dat = np.transpose(np.asarray(g_train))
    dat = np.transpose(np.asarray(np.concatenate((g_train, g_test))))

    # training parameters
    epsilon = 0.01
    momentum = 0.9
    batch_size = 64
    num_batches = dat.shape[1] // batch_size

    # model parameters
    num_vis = dat.shape[0]

    # initialize weights
    w_vh = 0.1 * np.random.randn(num_vis, num_hid)
    w_v = np.zeros((num_vis, 1))
    w_h = np.zeros((num_hid, 1))

    # initialize weight updates
    wu_vh = np.zeros((num_vis, num_hid))
    wu_v = np.zeros((num_vis, 1))
    wu_h = np.zeros((num_hid, 1))

    # Load if start epoch is different than 0
    # this piece of code loads the training persisted for previous iterations
    if start_epoch != 0:
        if start_epoch % 50 != 0:
            raise ValueError('start_epoch can only be multiple of 50')
        path = '../data/new_representation/rbm_' + str(num_hid) + '/_' + str(start_epoch)
        w_vh = np.load(path + '_numpy_w_vh_.npy')
        w_v = np.load(path + '_numpy_w_v_.npy')
        w_h = np.load(path + '_numpy_w_h_.npy')
        print('\n\nLoaded epoch: ' + str(start_epoch) + ' and error is ' + str(float(np.load(path + '_numpy_err_.npy')))
              + ' .....\n\n\n')

    start_time = time.time()
    for epoch in range(start_epoch + 1, end_epoch):
        print("Epoch %i" % (epoch + 1))
        print("Hidden units " + str(num_hid))
        err = []

        for batch in range(num_batches):
            v_true = dat[:, batch * batch_size:(batch + 1) * batch_size]
            v = v_true

            # apply momentum
            wu_vh *= momentum
            wu_v *= momentum
            wu_h *= momentum

            # positive phase
            h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

            wu_vh += np.dot(v, h.T)
            wu_v += v.sum(1)[:, np.newaxis]
            wu_h += h.sum(1)[:, np.newaxis]

            # sample hiddens
            wu_vh -= np.dot(v, h.T)
            wu_v -= v.sum(1)[:, np.newaxis]
            h = 1. * (h > np.random.rand(num_hid, batch_size))

            # negative phase
            v = 1. / (1 + np.exp(-(np.dot(w_vh, h) + w_v)))
            h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

            wu_h -= h.sum(1)[:, np.newaxis]

            # update weights
            w_vh += epsilon / batch_size * wu_vh
            w_v += epsilon / batch_size * wu_v
            w_h += epsilon / batch_size * wu_h

            err.append(np.mean((v - v_true) ** 2))

        print("Mean squared error: %f" % np.mean(err))
        print("Time: %f" % (time.time() - start_time))
        print('...........\n')

        # Dump training weights after every 50 iterations
        if epoch % 50 == 0:
            path = '../data/new_representation/rbm_' + str(num_hid) + '/_' + str(epoch)
            np.save(path + '_numpy_w_vh_', w_vh)
            np.save(path + '_numpy_w_v_', w_v)
            np.save(path + '_numpy_w_h_', w_h)
            np.save(path + '_numpy_err_', np.mean(err))


if __name__ == '__main__':
    train_rbm(num_hid=4000, start_epoch=1000, end_epoch=1101)
