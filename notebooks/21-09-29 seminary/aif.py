import sys
sys.path.insert(0, '../')

import numpy as np
from IPython.display import Markdown, display
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from collections import defaultdict
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dropout, Dense
from noisylabeltk.metrics import Accuracy
from aif360.algorithms import Transformer
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


np.random.seed(1)


label_map = {1.0: '>50K', 0.0: '<=50K'}
protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
ad = AdultDataset(protected_attribute_names=['sex'],
                  categorical_features=['workclass', 'education', 'marital-status',
                                        'occupation', 'relationship', 'native-country', 'race'],
                  privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                           'protected_attribute_maps': protected_attribute_maps})
(dataset_train,
 dataset_val,
 dataset_test) = AdultDataset().split([0.5, 0.8], shuffle=True)

sens_ind = 0
sens_attr = dataset_train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       dataset_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     dataset_train.privileged_protected_attributes[sens_ind]]


def describe(train=None, val=None, test=None):
    if train is not None:
        display(Markdown("#### Training Dataset shape"))
        print(train.features.shape)
    if val is not None:
        display(Markdown("#### Validation Dataset shape"))
        print(val.features.shape)
    display(Markdown("#### Test Dataset shape"))
    print(test.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(test.favorable_label, test.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(test.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(test.privileged_protected_attributes,
          test.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names"))
    print(test.feature_names)


def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())

    return metric_arrs


def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    plt.savefig('plot.png')

def forward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return:
    """
    P = K.constant(P)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

    return loss

class FairMLP(Transformer):

    def __init__(self, sensitive_attr='',
                 privileged_demotion=0.1, privileged_promotion=0.01,
                 protected_demotion=0.01, protected_promotion=0.1,
                 hidden_sizes=[32, 64, 32],
                 num_epochs=10, batch_size=64):
        self.p_list = [np.array([[1 - privileged_demotion, privileged_demotion],
                                 [privileged_promotion, 1 - privileged_promotion]]),
                       np.array([[1 - protected_demotion, protected_demotion],
                                [protected_promotion, 1 - protected_promotion]])]
        self.model = None
        self.hidden_sizes = hidden_sizes
        self.input_shape = None
        self.num_classes = 2
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.sensitive_attr = sensitive_attr
        self.classes_ =  np.array([0, 1])

    def _compile_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.input_shape))
        self.model.add(Dropout(0.2))

        for hidden_size in self.hidden_sizes:
            self.model.add(Dense(hidden_size, activation='relu'))

        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(optimizer='adam',
                    loss=self._build_loss(),
                    metrics=[Accuracy(name="Acc")])



    def _build_loss(self):

        loss_list = []
        n_sensitive = len(self.p_list)
        for P in self.p_list:
            loss_list.append(forward(P))

        def combined_loss(y_true, y_pred):
            n_classes = y_true.shape[1] - n_sensitive
            y_true_sensitive = y_true[:,:n_sensitive]
            y_true = y_true[:,-n_classes:]
            l0 = loss_list[0](y_true, y_pred) * y_true_sensitive[:,0]
            l1 = loss_list[1](y_true, y_pred) * y_true_sensitive[:,1]

            return l0+l1

        return combined_loss

    def fit(self, dataset):
        if self.model is None:
            self.input_shape = dataset.features.shape[1]
            self._compile_model()

        X = dataset.features
        y_expanded = np.zeros( shape=(X.shape[0], 4) )
        sensitive_index = dataset.protected_attribute_names.index(self.sensitive_attr)

        y_expanded[:,0] = (dataset.protected_attributes[:,sensitive_index] != dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,1] = (dataset.protected_attributes[:,sensitive_index] == dataset.unprivileged_protected_attributes[sensitive_index]).astype(int)
        y_expanded[:,2] = (dataset.labels == dataset.unfavorable_label).reshape(X.shape[0]).astype(int)
        y_expanded[:,3] = (dataset.labels == dataset.favorable_label).reshape(X.shape[0]).astype(int)

        self.model.summary()
        self.model.fit(X, y_expanded, verbose=True)

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        logits = self.predict_proba(X)
        return np.argmax(logits, axis=1)


describe(dataset_train, dataset_val, dataset_test)
dataset = dataset_train.copy()

fmlp = FairMLP(sensitive_attr='sex')
fmlp.fit(dataset)


thresh_arr = np.linspace(0.01, 0.5, 50)
val_metrics = test(dataset=dataset_val,
                   model=fmlp,
                   thresh_arr=thresh_arr)

lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])

disp_imp = np.array(val_metrics['disp_imp'])
disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     disp_imp_err, '1 - min(DI, 1/DI)')

#%%

plot(thresh_arr, 'Classification Thresholds',
     val_metrics['bal_acc'], 'Balanced Accuracy',
     val_metrics['avg_odds_diff'], 'avg. odds diff.')
