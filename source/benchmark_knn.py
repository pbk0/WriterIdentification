from data_handling import load_features_and_labels
from sklearn import neighbors
from sklearn.metrics import precision_recall_fscore_support

# global variables
g_train = None
g_train_label = None
g_test = None
g_test_label = None
g_feature_name = None


def main():

    # get data
    global g_train, g_train_label, g_test, g_test_label, g_feature_name
    g_train, g_train_label, g_test, g_test_label, g_feature_name = load_features_and_labels()

    # do knn
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(g_train, g_train_label)
    predictions = knn.predict(g_test)
    print knn.score(g_test, g_test_label)
    print precision_recall_fscore_support(g_test_label, predictions, average='micro')


if __name__ == "__main__":
    main()

