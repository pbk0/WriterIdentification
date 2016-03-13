
import unittest


class TestDataHandling(unittest.TestCase):
    """
    Class for unit testing functions in data_handling.py
    """

    # noinspection PyBroadException
    def test_load_features_and_labels(self):
        """
        test_load_features_and_labels
        :return:
        :rtype:
        """
        try:
            from .data_handling import load_features_and_labels
            _train, _train_label, _test, _test_label, _feature_name = load_features_and_labels()
        except Exception:
            self.fail('Failed to test_load_features_and_labels')