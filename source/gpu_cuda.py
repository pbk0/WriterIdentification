"""
https://github.com/EasonLiao/CudaTree

conda install wiserf
pip install pycuda
pip install cudatree
pip install parakeet
pip install nose
pip install dsltools
pip install pycuda --upgrade
"""

from data_handling import load_features_and_labels
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
from cudatree import RandomForestClassifier as cudaTreeRandomForestClassifier
from hybridforest import RandomForestClassifier as hybridForestRandomForestClassifier
from PyWiseRF import WiseRF


# global variables
g_train = None
g_train_label = None
g_test = None
g_test_label = None
g_feature_name = None


def main():

    do_it = 1

    # get data
    global g_train, g_train_label, g_test, g_test_label, g_feature_name
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()

    # do
    if do_it == 0:
        forest = cudaTreeRandomForestClassifier(n_estimators=50, verbose=True, bootstrap=False)
        forest.fit(np.asarray(g_train), np.asarray(g_train_label), bfs_threshold=4196)
        predictions = forest.predict(np.asarray(g_test))
        print precision_recall_fscore_support(g_test_label, predictions, average='micro')

    # do
    if do_it == 0:
        forest = hybridForestRandomForestClassifier(n_estimators=50,
                                                    n_gpus=2,
                                                    n_jobs=6,
                                                    bootstrap=False,
                                                    cpu_classifier=WiseRF)
        forest.fit(np.asarray(g_train), np.asarray(g_train_label), bfs_threshold=4196)
        predictions = forest.predict(np.asarray(g_test))
        print precision_recall_fscore_support(g_test_label, predictions, average='micro')


if __name__ == "__main__":
    main()

