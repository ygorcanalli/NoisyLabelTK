import numpy as np
from keras.metrics import BinaryAccuracy, Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Metric
from keras.callbacks import Callback
import tensorflow_addons as tfa


class FalseNegatives(FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(FalseNegatives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class FalsePositives(FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(FalsePositives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class TrueNegatives(TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(TrueNegatives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class TruePositives(TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(TruePositives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class Accuracy(BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(Accuracy, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class Precision(Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(Precision, self).update_state(y_true[:, -2:], y_pred, sample_weight)

class TruePositiveRate(Metric):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(Precision, self).update_state(y_true[:, -2:], y_pred, sample_weight)



def evaluate_discrimination(dataset_loader, model):
    dataset = dataset_loader.datasets[dataset_loader.name]()

    pred = model.predict(dataset['test']['features'])
    pred = np.argmax(pred, axis=1).reshape(pred.shape[0], 1)

    sensitive = (dataset['test']['sensitive'] == dataset['sensitive_positive_value']).to_numpy()
    accepted_positive = (pred & sensitive).sum()
    accepted_negative = (pred & ~sensitive).sum()

    total_positive = sensitive.sum()
    total_negative = (~sensitive).sum()

    return accepted_positive / total_positive - accepted_negative / total_negative

def BinaryMCC(name):
    return tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, name=name)

