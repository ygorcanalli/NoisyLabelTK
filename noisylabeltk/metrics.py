import numpy as np
from keras.metrics import BinaryAccuracy, Precision, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Metric
from keras.callbacks import Callback
import tensorflow_addons as tfa
from pprint import pprint

class Accuracy(BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super(Accuracy, self).update_state(y_true[:, -2:], y_pred, sample_weight)

def fairness_metrics_from_confusion_matrix(tp, tn, fp, fn):

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    ppv = tp / (tp + fp)
    fdr = fp / (fp + tp)
    npv = tn / (tn + fn)
    FOR = fn / (fn + tn)
    positives = (tp + fp) / (tp + fp + tn + fn)

    rates = {
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'PPV': ppv,
        'FDR': fdr,
        'NPV': npv,
        'FOR': FOR,
        'Positives': positives
    }

    return rates

def BinaryMCC(name):
    return tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, name=name)

