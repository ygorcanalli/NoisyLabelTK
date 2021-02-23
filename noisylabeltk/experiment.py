# %%
import tensorflow as tf
from noisylabeltk.datasets import DatasetLoader
from noisylabeltk.loss import cross_entropy, boot_soft
import noisylabeltk.models as models
import neptune
import datetime
import numpy as np
from neptunecontrib.monitoring.keras import NeptuneMonitor

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
                                             upload_source_files=['main.py'])
        self.parameters['transition-matrix'] = np.array(self.parameters['transition-matrix'])
        self.train = None
        self.validation = None
        self.test = None
        self.num_features = None
        self.num_classes = None

    def _load_data(self):

        dataset_loader = DatasetLoader(self.parameters['dataset'])

        if 'robust-method' in self.parameters and self.parameters['robust-method'] is not None and \
                self.parameters['robust-method'] != 'none' and 'transition_matrix' in self.parameters:
            self.exp.set_property('transition_matrix', self.parameters['transition_matrix'])
            (train_ds, validation_ds, test_ds), num_features, num_classes = dataset_loader.pollute_and_load(
                self.parameters['transition-matrix'])
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
        self.model = models.create_model(self.parameters['model'], self.num_features, self.num_classes)
        self.model.compile(optimizer='adam',
                           loss=cross_entropy,
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
        self._load_data()
        self._build_model()
        self._fit_model()
        self._evaluate()
        self.exp.stop()
