from noisylabeltk.loss import make_loss
from datetime import datetime
from noisylabeltk.metrics import Accuracy, fairness_metrics_from_confusion_matrix, evaluate_auc


import noisylabeltk.models as models
import numpy as np
from pprint import pformat
from tensorflow.keras.callbacks import EarlyStopping

class Experiment(object):

    def __init__(self, num_features, num_classes, parameters, database_name, collection_name=None, tags=None):

        self.parameters = parameters
        self.database_name = database_name
        self.collection_name = collection_name
        self.run_entry = dict()
        self.run_entry['begin'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self.run_entry['tags'] = tags



        if 'noise_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['noise-args']):
                self.run_entry['parameters']['noise_arg_%d' % i] = arg

        if 'loss_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['loss-args']):
                self.run_entry['parameters']['loss_arg_%d' % i] = arg

        if 'loss_kwargs' in self.parameters and self.parameters is not None:
            for key, value in enumerate(self.parameters['loss-kwargs']):
                self.run_entry['parameters']['loss_arg_%d' % key] = value

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
        self.run_entry['metrics'] = dict()
        self.run_entry['parameters'] = self.parameters
        self.run_entry['parameters']['num-features'] = self.num_features
        self.run_entry['parameters']['num-classes'] = self.num_classes
        self.run_entry['parameters']['loss-args'] = pformat(self.parameters['loss-args'])
        self.model = None


    def build_model(self, hyperparameters):

        self.run_entry['hyperparameters'] = hyperparameters

        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes, **hyperparameters)
        self.model.compile(optimizer='adam',
                           loss=self.loss_function,
                           metrics=[Accuracy(name="Acc")])

        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        self.run_entry['parameters']['model-summary'] = '\n'.join(summary_list)

    def fit_model(self, train, verbose=False):

        callbacks = []
        if 'patience' in self.parameters:
            es = EarlyStopping(monitor='loss', patience=self.parameters['patience'])
            callbacks.append(es)

        history = self.model.fit(train['features'], train['labels'], self.parameters['batch-size'],
                                 callbacks=callbacks,
                                 epochs=self.parameters['epochs'],
                                 validation_split=0.2,
                                 verbose=verbose)





    def evaluate(self, test, prefix='test', verbose=False):
        eval_metrics = self.model.evaluate(test['features'], test['labels'], verbose=verbose)

        for j, metric in enumerate(eval_metrics):
            self.run_entry['metrics'][prefix + '_' + self.model.metrics_names[j]] = metric

    def evaluate_discrimination(self, test, prefix='test'):

        pred = self.model.predict(test['features'])
        logits = np.asarray(pred[:,1])
        pred = np.asarray(np.argmax(pred, axis=1)).reshape(pred.shape[0]).astype(bool)
        true = np.asarray(np.argmax(test['labels'][:,-2:], axis=1)).reshape(pred.shape[0]).astype(bool)

        protected = np.asarray(test['labels'][:, 1]).reshape(pred.shape[0]).astype(bool)

        protected_tp = (pred & true & protected).sum()
        protected_tn = (~pred & ~true & protected).sum()
        protected_fp = (pred & ~true & protected).sum()
        protected_fn = (~pred & true & protected).sum()

        privileged_tp = (pred & true & ~protected).sum()
        privileged_tn = (~pred & ~true & ~protected).sum()
        privileged_fp = (pred & ~true & ~protected).sum()
        privileged_fn = (~pred & true & ~protected).sum()

        overall_tp = (pred & true).sum()
        overall_tn = (~pred & ~true).sum()
        overall_fp = (pred & ~true).sum()
        overall_fn = (~pred & true).sum()

        protected_rates = fairness_metrics_from_confusion_matrix(protected_tp, protected_tn,
                                                                 protected_fp, protected_fn)

        privileged_rates = fairness_metrics_from_confusion_matrix(privileged_tp, privileged_tn,
                                                                  privileged_fp, privileged_fn)

        overall_rates = fairness_metrics_from_confusion_matrix(overall_tp, overall_tn,
                                                               overall_fp, overall_fn)


        try:
            auc = evaluate_auc(true.astype(int), logits)
        except:
            print(true.astype(int))
            print(logits)

        self.run_entry['metrics']['%s_AUC_overall' % prefix] = auc

        for name in privileged_rates.keys():
            privileged_value = privileged_rates[name]
            protected_value = protected_rates[name]
            overall_value = overall_rates[name]
            self.run_entry['metrics']['%s_%s_protected' % (prefix, name)] = protected_value
            self.run_entry['metrics']['%s_%s_privileged' % (prefix, name)] = privileged_value
            self.run_entry['metrics']['%s_%s_overall' % (prefix, name)] = overall_value
            self.run_entry['metrics']['%s_%s_balance' % (prefix, name)] = privileged_value - protected_value
            self.run_entry['metrics']['%s_%s_relative_balance' % (prefix, name)] = (privileged_value - protected_value) / overall_value



        audit_acc = (pred == true).sum() / test['labels'].shape[0]
        self.run_entry['metrics']['%s_audit_acc' % prefix] = audit_acc

    def persist_metadata(self):

        self.run_entry['end'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        client = MongoClient('localhost', 27017)
        db = client[self.database_name]
        collection = db[self.collection_name]
        collection.insert_one(self.run_entry)
