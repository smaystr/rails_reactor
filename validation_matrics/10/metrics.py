from sklearn import metrics
import numpy as np


class Metrics:

    def _verify_length(self, y_true, y_pred):
        assert y_true.shape[0] > 0, 'y_true length must be at least 1'
        assert y_pred.shape[0] > 0, 'y_pred length must be at least 1'
        assert y_true.shape[0] == y_pred.shape[0], 'y_true and y_pred must have the same length'

    def mean_squared_error(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return np.mean(np.square(y_true - y_pred))

    def mean_squared_log_error(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return np.mean(np.square((np.log(y_true + 1) - np.log(y_pred + 1))))

    def mean_absolute_error(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return np.mean(np.abs(y_true - y_pred))

    def median_absolute_error(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return np.median(np.abs(y_true - y_pred))

    def r2_score(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))

    def explained_variance_score(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        y_error = y_true - y_pred
        return 1 - np.sum(np.square(y_error - np.mean(y_error))) / np.sum(np.square(y_true - np.mean(y_true)))

    def max_error(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)

        return metrics.max_error(y_true, y_pred)

    def get_prediction_types(self, y_true, y_pred):
        true_positive = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        false_positive = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))

        return true_positive, false_positive, false_negative

    def recall_score(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)
        tp, fp, fn = self.get_prediction_types(y_true, y_pred)

        return tp / (tp + fn)

    def precision_score(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)
        tp, fp, fn = self.get_prediction_types(y_true, y_pred)

        return tp / (tp + fp)

    def fbeta_score(self, y_true, y_pred, beta):
        self._verify_length(y_true, y_pred)
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)

        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    def f1_score(self, y_true, y_pred):
        self._verify_length(y_true, y_pred)
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)

        return 2 * precision * recall / (precision + recall)
