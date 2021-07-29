import numpy as np
from keras.metrics import BinaryAccuracy, Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
import tensorflow_addons as tfa


class CustomFalseNegatives(FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomFalseNegatives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class CustomFalsePositives(FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomFalsePositives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class CustomTrueNegatives(TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomTrueNegatives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class CustomTruePositives(TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomTruePositives, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class CustomAccuracy(BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomAccuracy, self).update_state(y_true[:, -2:], y_pred, sample_weight)


class CustomPrecision(Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(CustomPrecision, self).update_state(y_true[:, -2:], y_pred, sample_weight)


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

def BinaryMCC():
    return tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
