# %%
import sys
import traceback
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.loss import make_loss
import noisylabeltk.models as models
import neptune
import datetime
import numpy as np
from neptunecontrib.monitoring.keras import NeptuneMonitor
from git import Repo
import io
import os

PROJECT_NAME = 'ygorcanalli/NoisyLabelTK'


class Experiment(object):

    def __init__(self, parameters):

        self.parameters = parameters
        self.name = "%s-%s-%s" % (
            self.parameters['dataset'], self.parameters['noise'], self.parameters['robust-method'])

        self.description = "Label Noise Robustness"

        neptune.init(project_qualified_name=PROJECT_NAME)
        self.exp = neptune.create_experiment(name=self.name,
                                             description=self.description,
                                             tags=[self.parameters['dataset'], self.parameters['robust-method']],
                                             params=parameters,
                                             upload_source_files=['noiselabeltk/main.py'])

        if 'noise_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['noise_args']):
                self.exp.set_property('noise_arg_%d' % i, arg)

        if 'loss_args' in self.parameters and self.parameters is not None:
            for i, arg in enumerate(self.parameters['loss_args']):
                self.exp.set_property('loss_arg_%d' % i, arg)

        if 'loss_kwargs' in self.parameters and self.parameters is not None:
            for key, value in enumerate(self.parameters['loss_kwargs']):
                self.exp.set_property('loss_arg_%s' % key, value)

        self.train = None
        self.validation = None
        self.test = None
        self.num_features = None
        self.num_classes = None

        repo = Repo(os.getcwd())
        t = repo.head.commit.tree
        diff_string = repo.git.diff(t)
        data_io = io.StringIO(diff_string)

        self.exp.log_artifact(data_io, destination='git_diff.txt')
        data_io.close()

    def _load_data(self):

        dataset_loader = DatasetLoader(self.parameters['dataset'])

        if 'noise' in self.parameters and self.parameters['noise'] is not None and \
                self.parameters['noise'] != 'none':
            args = []
            if 'noise-args' in self.parameters and self.parameters['noise-args'] is not None:
                args = self.parameters['noise-args']

            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.pollute_and_load(self.parameters['noise'], *args)
        else:
            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.load()

        self.train = train_ds
        self.validation = validation_ds
        self.test = test_ds
        self.num_features = num_features
        self.num_classes = num_classes

        self.exp.set_property('num_features', num_features)
        self.exp.set_property('num_classes', num_classes)

    def _build_model(self):

        if 'robust-method' in self.parameters and self.parameters['robust-method'] is not None and \
                self.parameters['robust-method'] != 'none':
            args = []
            kwargs = {}
            if 'loss-args' in self.parameters and self.parameters['loss-args'] is not None:
                args = self.parameters['loss-args']
            if 'loss-kwargs' in self.parameters and self.parameters['loss-kwargs'] is not None:
                kwargs = self.parameters['loss-kwargs']

            loss_function = make_loss(self.parameters['robust-method'], *args, **kwargs)
        else:
            loss_function = make_loss('cross-entropy')

        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes)
        self.model.compile(optimizer='adam',
                           loss=loss_function,
                           metrics=['accuracy'])

        self.model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

    def _fit_model(self):
        history = self.model.fit(self.train, epochs=10,
                                 validation_data=self.validation,
                                 callbacks=[NeptuneMonitor()])

    def _evaluate(self):
        eval_metrics = self.model.evaluate(self.test, verbose=0)

        for j, metric in enumerate(eval_metrics):
            neptune.log_metric('eval_' + self.model.metrics_names[j], metric)

    def run(self):
        try:
            self._load_data()
            self._build_model()
            self._fit_model()
            self._evaluate()
        except Exception as ex:
            try:
                exc_info = sys.exc_info()
            finally:
                # Display the *original* exception
                traceback.print_exception(*exc_info)
                self.exp.stop('\n'.join(traceback.format_exception(*exc_info)))
                del exc_info
        else:
            self.exp.stop()
