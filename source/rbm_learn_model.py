"""

"""

from __future__ import division
from data_handling import load_features_and_labels
import time
import numpy as np
#import util

# load data
g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()
#util.load('mnist.dat', globals())
#dat = dat/255.
dat = np.transpose(np.asarray(g_train))

# training parameters
epsilon = 0.01
momentum = 0.9

num_epochs = 10001
batch_size = 64
num_batches = dat.shape[1]//batch_size

# model parameters
num_vis = dat.shape[0]
num_hid = 4000


# initialize weights
w_vh = 0.1 * np.random.randn(num_vis, num_hid)
w_v = np.zeros((num_vis, 1))
w_h = np.zeros((num_hid, 1))

# initialize weight updates
wu_vh = np.zeros((num_vis, num_hid))
wu_v = np.zeros((num_vis, 1))
wu_h = np.zeros((num_hid, 1))

start_time = time.time()
for epoch in range(num_epochs):
    print("Epoch %i" % (epoch + 1))
    print("Hidden units " + str(num_hid))
    err = []

    for batch in range(num_batches):
        v_true = dat[:, batch*batch_size:(batch + 1)*batch_size]
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
        h = 1. * (h > np.random.rand(num_hid, batch_size))

        # negative phase
        v = 1. / (1 + np.exp(-(np.dot(w_vh, h) + w_v)))
        h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

        wu_vh -= np.dot(v, h.T)
        wu_v -= v.sum(1)[:, np.newaxis]
        wu_h -= h.sum(1)[:, np.newaxis]

        # update weights
        w_vh += epsilon/batch_size * wu_vh
        w_v += epsilon/batch_size * wu_v
        w_h += epsilon/batch_size * wu_h

        err.append(np.mean((v - v_true)**2))

    print("Mean squared error: %f" % np.mean(err))
    print("Time: %f" % (time.time() - start_time))

    if epoch % 100 == 0:
        np.save('rbm_numpy/_' + str(epoch) + '_numpy_w_vh_', w_vh)
        np.save('rbm_numpy/_' + str(epoch) + '_numpy_w_v_', w_v)
        np.save('rbm_numpy/_' + str(epoch) + '_numpy_w_h_', w_h)
        np.save('rbm_numpy/_' + str(epoch) + '_numpy_err_', np.mean(err))


