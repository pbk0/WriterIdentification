from data_handling import load_features_and_labels
from sklearn import svm
import numpy as np

# global variables
g_train = None
g_train_label = None
g_test = None
g_test_label = None
g_feature_name = None


def dot_product_kernel(x, y):
    return np.dot(x, y.T)


def main():

    do_it = 1

    # get data
    global g_train, g_train_label, g_test, g_test_label, g_feature_name
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()

    # do svm one vs one
    if do_it == 1:
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(g_train, g_train_label)
        print clf.score(g_test, g_test_label)

    # do svm one vs one and kernel='linear'
    if do_it == 1:
        clf1 = svm.SVC(decision_function_shape='ovo', kernel='linear')
        clf1.fit(g_train, g_train_label)
        print clf1.score(g_test, g_test_label)

    # do svm one vs one and kernel='linear'
    if do_it == 1:
        clf2 = svm.SVC(decision_function_shape='ovo', kernel='rbf')
        clf2.fit(g_train, g_train_label)
        print clf2.score(g_test, g_test_label)

    # do svm one vs one and kernel=dot_product_kernel
    if do_it == 1:
        clf3 = svm.SVC(decision_function_shape='ovo', kernel=dot_product_kernel)
        clf3.fit(g_train, g_train_label)
        print clf3.score(g_test, g_test_label)

    # do linearSVM one vs rest
    if do_it == 1:
        lin_clf = svm.LinearSVC()
        lin_clf.fit(g_train, g_train_label)
        print lin_clf.score(g_test, g_test_label)

    # TODO: try LDA with sklearn


if __name__ == "__main__":
    main()

