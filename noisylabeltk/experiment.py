#from noisylabeltk.seed import ensure_seterministic, get_seed
from tensorflow.keras.losses import categorical_crossentropy
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.loss import make_loss
from noisylabeltk.metrics import Accuracy, TruePositives, TrueNegatives,\
                                    FalsePositives, FalseNegatives, BinaryMCC, \
                                    statistical_parity, fairness_metrics_from_confusion_matrix

import noisylabeltk.models as models
import neptune.new as neptune
import optuna
from optuna.samplers import TPESampler
from neptune.new.integrations.tensorflow_keras import NeptuneCallback as NeptuneKerasCallback
from neptunecontrib.monitoring.optuna import NeptuneCallback as NeptuneOptunaCallback
import numpy as np

acc = Accuracy(name="Acc")
mcc = BinaryMCC(name="MCC")

class Experiment(object):

    def __init__(self, num_features, num_classes, parameters, project_name, experiment_name, tags=None):

        self.parameters = parameters
        self.project_name = project_name
        self.name = experiment_name
        self.tags = tags

        if 'noise_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['noise-args']):
                self.run['parameters/noise_arg_%d' % i] = arg

        if 'loss_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['loss-args']):
                self.run['parameters/loss_arg_%d' % i] = arg

        if 'loss_kwargs' in self.parameters and self.parameters is not None:
            for key, value in enumerate(self.parameters['loss-kwargs']):
                self.run['parameters/loss_arg_%d' % key] = value

        if 'robust-method' in self.parameters and self.parameters['robust-method'] is not None and \
                self.parameters['robust-method'] != 'none':
            args = []
            kwargs = {}
            if 'loss-args' in self.parameters and self.parameters['loss-args'] is not None:
                args = self.parameters['loss-args']
            if 'loss-kwargs' in self.parameters and self.parameters['loss-kwargs'] is not None:
                kwargs = self.parameters['loss-kwargs']

            self.loss_function = make_loss(self.parameters['robust-method'], *args, **kwargs)
        else:
            self.loss_function = make_loss('cross-entropy')

        self.num_features = num_features
        self.num_classes = num_classes
        self.neptune_run = None
        self.model = None

    def build_model(self, hyperparameters):

        if self.neptune_run is None:
            self._init_tracking()

        self.neptune_run['parameters/hyperparameters'] = hyperparameters

        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes, **hyperparameters)
        self.model.compile(optimizer='adam',
                           loss=self.loss_function,
                           metrics=[acc, mcc])

        self.model.summary(print_fn=lambda x: self.neptune_run['model_summary'].log(x))

    def fit_model(self, train, validation, batch_size):

        neptune_cbk = NeptuneKerasCallback(self.neptune_run, base_namespace='metrics')
        history = self.model.fit(train['features'], train['labels'], batch_size, epochs=10,
                                 validation_data=(validation['features'], validation['labels']),
                                 callbacks=[neptune_cbk],
                                 verbose=0)


    def evaluate(self, test, batch_size):
        eval_metrics = self.model.evaluate(test['features'], test['labels'], batch_size, verbose=0)

        for j, metric in enumerate(eval_metrics):
            self.neptune_run['metrics/eval_' + self.model.metrics_names[j]].log(metric)

    def _confusion_matrix_from_metrics(self, metrics):
        for j, metric in enumerate(metrics):
            if self.model.metrics_names[j] == 'TP':
                tp = metric
            elif self.model.metrics_names[j] == 'TN':
                tn = metric
            elif self.model.metrics_names[j] == 'FP':
                fp = metric
            elif self.model.metrics_names[j] == 'FN':
                fn = metric

        return tp, tn, fp, fn

    def evaluate_discrimination(self, test, batch_size):

        protected = np.squeeze(np.asarray(test['labels'][:, 1] == 1))
        unprotected = np.squeeze(np.asarray(test['labels'][:, 0] == 1))

        unprotected_metrics = self.model.evaluate(test['features'][unprotected], test['labels'][unprotected], \
                                                  batch_size=batch_size, verbose=0)
        protected_metrics = self.model.evaluate(test['features'][protected], test['labels'][protected], \
                                                  batch_size=batch_size, verbose=0)

        for j, (un_metric, prot_metric) in enumerate(zip(unprotected_metrics, protected_metrics)):

            self.neptune_run['metrics/eval_' + self.model.metrics_names[j] + "_balance"].log(un_metric - prot_metric)

        tp, tn, fp, fn = self._confusion_matrix_from_metrics(unprotected_metrics)
        un_rates = fairness_metrics_from_confusion_matrix(tp, tn, fp, fn)
        tp, tn, fp, fn = self._confusion_matrix_from_metrics(protected_metrics)
        prot_rates = fairness_metrics_from_confusion_matrix(tp, tn, fp, fn)

        for name in un_rates.keys():
            unprotected_value = un_rates[name]
            protected_value = prot_rates[name]
            self.neptune_run['metrics/eval_%s_balance' % name].log(unprotected_value - protected_value)

        parity = statistical_parity(test, self.model)
        self.neptune_run['metrics/eval_statistical_parity'].log(parity)

    def evaluate_discrimination_2(self, test, batch_size):

        pred = self.model.predict(test['features'])
        pred = np.argmax(pred, axis=1).reshape(pred.shape[0], 1).astype(bool)

        true = np.argmax(test['labels'][:,-2:], axis=1).reshape(pred.shape[0], 1).astype(bool)

        protected = test['labels'][:, 1].astype(bool)

        protected_tp = (pred & true & protected).sum()
        protected_tn = (~pred & ~true & protected).sum()
        protected_fp = (pred & ~true & protected).sum()
        protected_fn = (~pred & true & protected).sum()

        unprotected_tp = (pred & true & ~protected).sum()
        unprotected_tn = (~pred & ~true & ~protected).sum()
        unprotected_fp = (pred & ~true & ~protected).sum()
        unprotected_fn = (~pred & true & ~protected).sum()

        un_rates = fairness_metrics_from_confusion_matrix(protected_tp, protected_tn, \
                                                          protected_fp, protected_fn)

        prot_rates = fairness_metrics_from_confusion_matrix(unprotected_tp, unprotected_tn, \
                                                            unprotected_fp, unprotected_fn)

        for name in un_rates.keys():
            unprotected_value = un_rates[name]
            protected_value = prot_rates[name]
            self.neptune_run['metrics/eval_%s_balance' % name].log(unprotected_value - protected_value)

    def _init_tracking(self):
        self.neptune_run = neptune.init(project=self.project_name,
                                        mode = "async",
                                        tags=self.tags,
                                        name=self.name,
                                        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZGUxY2IwMy1kOTMzLTRjMTUtYjAxYy01MWE2MmMyYzQ0ZmYifQ==")

        self.neptune_run["model/params"] = self.parameters
        self.neptune_run['parameters/num-features'] = self.num_features
        self.neptune_run['parameters/num-classes'] = self.num_classes

    def stop_tracking(self):
        self.neptune_run.stop()
