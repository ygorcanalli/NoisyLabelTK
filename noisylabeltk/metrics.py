import numpy as np
from keras.metrics import BinaryAccuracy

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
    negatives = (tn + fn) / (tp + fp + tn + fn)
    MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ACC = (tp + tn) / (tp + fp + tn + fn)

    rates = {
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'PPV': ppv,
        'FDR': fdr,
        'NPV': npv,
        'FOR': FOR,
        'MCC': MCC,
        'ACC': ACC,
        'Positives': positives,
        'Negatives': negatives
    }

    return rates
