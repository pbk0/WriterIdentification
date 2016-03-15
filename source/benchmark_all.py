
from data_handling import load_features_and_labels
from rbm_generate_new_representation import load_rbm_model_features_and_labels
from joblib import Parallel, delayed
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

# global variables
g_train = None
g_train_label = None
g_test = None
g_test_label = None
g_feature_name = None
g_num_cores = multiprocessing.cpu_count()
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
g_classifiers_try = {
    #"Pipe SVC LDA": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', LinearDiscriminantAnalysis())]),
    #"Linear Discriminant Analysis svd": LinearDiscriminantAnalysis(),
    "Extra Trees 1000": ExtraTreesClassifier(n_estimators=1000),
    #"Pipe SVC Extra Trees 1000": Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', ExtraTreesClassifier(n_estimators=1000))]),
    #"1 Nearest Neighbors": KNeighborsClassifier(1),
    #"SVM Linear": SVC(kernel="linear", C=0.025),
    #"SVM RBF": SVC(gamma=2, C=1),
}


def execute_classifier(classifier_name, classifier):
    print("...................................................................")
    start_time = time.time()
    classifier.fit(g_train, g_train_label)
    train_predictions = classifier.predict(g_train)
    test_predictions = classifier.predict(g_test)
    train_score = f1_score(g_train_label, train_predictions, average='micro')
    test_score = f1_score(g_test_label, test_predictions, average='micro')
    end_time = time.time()

    print_str = "\nClassifier: " + classifier_name + \
                "\n\t-- train score: " + str(train_score) + \
                "\n\t-- test score : " + str(test_score)

    if True:
        if type(classifier) is ExtraTreesClassifier or type(classifier) is RandomForestClassifier:
            top_10_feat = np.argsort(classifier.feature_importances_)
            print_str += "\n\t-- top 100 features: "
            index = 0
            for i in top_10_feat:
                index += 1
                print_str += "\n\t\t - " + str(index) + "\t" + str(i) + "\t" + g_feature_name[i]

    print_str += "\n\t-- time taken :" + str(end_time-start_time)

    print(print_str)
    f_out = open("report/_Classifier " + classifier_name+".txt", "w")
    f_out.write(print_str + "\n")
    f_out.close()
    return print_str


def main():

    # get data
    global g_train, g_train_label, g_test, g_test_label, g_feature_name, g_num_cores, g_classifiers
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()
    #g_train, g_train_label, g_test, g_test_label, g_feature_name = load_rbm_model_features_and_labels()

    results = Parallel(n_jobs=g_num_cores)(delayed(execute_classifier)(k, v) for k, v in g_classifiers_try.iteritems())

    f_out = open("report/report_pk.txt", "w")
    for line in results:
        f_out.write(line + "\n")
    f_out.close()


if __name__ == "__main__":
    main()

