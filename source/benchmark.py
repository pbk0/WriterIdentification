"""
This file contains code for benchmarking the scores on test data-set
"""
from source.utility.data_handling import load_features_and_labels
from source.generate_new_representation import rbm_representation
from joblib import Parallel, delayed
import gc
import multiprocessing
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
from collections import namedtuple

# global variables
g_train = None
g_train_label = None
g_test = None
g_test_label = None
g_feature_name = None
g_num_cores = multiprocessing.cpu_count()

# special memory structure to be used by multi threaded code
parallel_mem_struct = namedtuple(
    'parallel_mem_struct',
    [
        'classifier_title',
        'new_representation',
        'classifier',
        'time_taken',
        'train_score',
        'test_score',
        'top_features'
    ]
)

g_classifiers = {
    #"Logistic Regression": LogisticRegression(),
    #"SGD": SGDClassifier(),
    #"Gaussian Naive Bayes": GaussianNB(),
    "1 Nearest Neighbors": KNeighborsClassifier(1),
    #"2 Nearest Neighbors": KNeighborsClassifier(2),
    #"3 Nearest Neighbors": KNeighborsClassifier(3),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "SVM Linear": SVC(kernel="linear", C=0.025),
    "SVM RBF": SVC(gamma=2, C=1),
    "Decision Tree": DecisionTreeClassifier(),
    #"AdaBoost 10": AdaBoostClassifier(n_estimators=10),
    #"AdaBoost 100": AdaBoostClassifier(n_estimators=100),
    #"AdaBoost 1000": AdaBoostClassifier(n_estimators=1000),
    "Random Forest 10": RandomForestClassifier(n_estimators=10),
    "Random Forest 100": RandomForestClassifier(n_estimators=100),
    "Random Forest 1000": RandomForestClassifier(n_estimators=1000),
    "Random Forest 10000": RandomForestClassifier(n_estimators=10000),
    "Extra Trees 10": ExtraTreesClassifier(n_estimators=10),
    "Extra Trees 100": ExtraTreesClassifier(n_estimators=100),
    "Extra Trees 1000": ExtraTreesClassifier(n_estimators=1000),
    "Extra Trees 10000": ExtraTreesClassifier(n_estimators=10000),
    #"Gradient Boosting 10": GradientBoostingClassifier(n_estimators=10),
    #"Gradient Boosting 100": GradientBoostingClassifier(n_estimators=100),
    #"Gradient Boosting 1000": GradientBoostingClassifier(n_estimators=1000),
    "Pipe SVC LDA": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', LinearDiscriminantAnalysis())]),
    "Pipe SVC Extra Trees 1000": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', ExtraTreesClassifier(n_estimators=1000))]),
}


def get_classifier_dict():
    """
    This method creates couple of classifiers that will be used for training.
    This is the place where you decide which classifiers to use.
    Populate the dictionary accordingly.
    :return: dictionary
    :rtype: dict
    """
    dictionary = {
        "Logistic Regression": LogisticRegression(),
        "SGD": SGDClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "1 Nearest Neighbors": KNeighborsClassifier(1),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "SVM Linear": SVC(kernel="linear", C=0.025),
        "SVM RBF": SVC(gamma=2, C=1),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest 10": RandomForestClassifier(n_estimators=10),
        "Random Forest 100": RandomForestClassifier(n_estimators=100),
        "Random Forest 1000": RandomForestClassifier(n_estimators=1000),
        "Extra Trees 10": ExtraTreesClassifier(n_estimators=10),
        "Extra Trees 100": ExtraTreesClassifier(n_estimators=100),
        "Extra Trees 1000": ExtraTreesClassifier(n_estimators=1000),
        "Pipe SVC LDA": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', LinearDiscriminantAnalysis())]),
        "Pipe SVC Extra Trees 1000": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', ExtraTreesClassifier(n_estimators=1000))]),
    }
    return dictionary


def get_parallel_mem_struct(classifier_title, classifier, new_representation):
    """
    Create a namedtuple interface using parallel_mem_struct to store results from different threads.
    This structure will provide clean interface to store results
    :param classifier_title: title for classifier
    :type classifier_title: str
    :param classifier:
    :type classifier:
    :param new_representation: new representation to be used ['None', 'RBM', 'PCA']
    :type new_representation: str
    :return: namedtuple parallel_mem_struct
    :rtype: parallel_mem_struct
    """
    return parallel_mem_struct(
        classifier_title=classifier_title,
        classifier=classifier,
        new_representation=new_representation
    )


def execute_classifier(parallel_mem_struct_):
    """
    Special generic method that can take any classifier and execute it.
    It also reports the score on provided dataset.
    This method is specially designed to exploit multiprocessor usage in trying different classifiers independently.
    :param parallel_mem_struct_: generic struct so that multiple threads can use it
    :type parallel_mem_struct_:
    :return:
    :rtype:
    """
    start_time = time.time()
    parallel_mem_struct_.classifier.fit(g_train, g_train_label)
    train_predictions = parallel_mem_struct_.classifier.predict(g_train)
    test_predictions = parallel_mem_struct_.classifier.predict(g_test)
    parallel_mem_struct_.train_score = f1_score(g_train_label, train_predictions, average='micro')
    parallel_mem_struct_.test_score = f1_score(g_test_label, test_predictions, average='micro')
    end_time = time.time()
    parallel_mem_struct_.time_taken = end_time - start_time

    if True:
        top_features = ''
        if type(parallel_mem_struct_.classifier) is ExtraTreesClassifier \
                or type(parallel_mem_struct_.classifier) is RandomForestClassifier:
            top_feat = np.argsort(parallel_mem_struct_.classifier.feature_importances_)
            for i in top_feat:
                top_features += str(i) + ' '
            parallel_mem_struct_.top_features = top_features
        else:
            parallel_mem_struct_.top_features = top_features
                
    return parallel_mem_struct_
