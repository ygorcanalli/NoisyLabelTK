from noisylabeltk.util import set_seed, get_seed

set_seed(777)

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from pprint import pprint
from noisylabeltk.noise_generator import build_noise_generator

BASE_PATH = 'datasets'

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds

def np_to_dataset(features, labels, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
        ds = ds.batch(batch_size)
    return ds


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


class DatasetLoader(object):

    def __init__(self, name, batch_size, sensitive_labels=False):
        self.name =  name
        self.sensitive_labels=sensitive_labels
        self.datasets = {
            'income': self._load_income,
            'german': self._load_germancredit,
            'german_sex': self._load_germancredit,
            'german_age': self._load_germancredit_age,
        }
        self.batch_size = batch_size

    def load(self):
        dataset_loader = self.datasets[self.name]
        dataset = dataset_loader()
        return dataset

    def pollute_and_load(self, noise_name, *args):
        dataset_loader = self.datasets[self.name]
        dataset = dataset_loader()

        noise_generator = build_noise_generator(noise_name, dataset['train']['labels'], *args)
        noisy_train_labels = noise_generator.generate_noisy_labels()

        noise_generator = build_noise_generator(noise_name, dataset['validation']['labels'], *args)
        noisy_validation_labels = noise_generator.generate_noisy_labels()

        train_ds = np_to_dataset(dataset['train']['features'], noisy_train_labels, batch_size=self.batch_size)
        validation_ds = np_to_dataset(dataset['validation']['features'], noisy_validation_labels, batch_size=self.batch_size)
        test_ds = np_to_dataset(dataset['test']['features'], dataset['test']['labels'], batch_size=self.batch_size)

        return (train_ds, validation_ds, test_ds), dataset['num_features'], dataset['num_classes']

    def _load_germancredit(self):
        path = os.path.join(BASE_PATH, 'german_credit', 'german.data')
        column_names = ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employmentsince',
                        'installmentate', 'personalstatus', 'garantors', 'residencesince', 'property', 'age',
                        'othereinstallments', 'housing', 'existingcredits', 'job', 'numbermaintence', 'telephone',
                        'foreign', 'label']

        dataframe = pd.read_csv(path, names=column_names, delim_whitespace=True)
        categorical_features = ['status', 'history', 'purpose', 'savings', 'employmentsince', 'personalstatus',
                                'garantors', 'property', 'othereinstallments', 'housing', 'job', 'telephone']
        numerical_features = ['duration', 'amount', 'installmentate', 'residencesince', 'age', 'existingcredits',
                              'numbermaintence']

        dataframe['label'] = dataframe['label'].astype(str)
        dataframe.loc[dataframe['label'] == '1','label'] = 'Good'
        dataframe.loc[dataframe['label'] == '2','label'] = 'Bad'

        # A91: male: divorced / separated
        # A92: female: divorced / separated / married
        # A93: male: single
        # A94: male: married / widowed
        # A95: female: single

        sensitive_features = ['personalstatus']
        dataframe.loc[dataframe['personalstatus'] == 'A91', 'personalstatus'] = 'Male'
        dataframe.loc[dataframe['personalstatus'] == 'A93', 'personalstatus'] = 'Male'
        dataframe.loc[dataframe['personalstatus'] == 'A94', 'personalstatus'] = 'Male'
        dataframe.loc[dataframe['personalstatus'] == 'A92', 'personalstatus'] = 'Female'
        #dataframe.loc[dataframe['personalstatus'] == 'A95', 'personalstatus'] = 'Female' no single female in dataset
        privileged_value = 'Male'
        protected_value = 'Female'

        target_class = ['label']
        negative_class = 'Bad'
        positive_class = 'Good'

        ct = ColumnTransformer([
            ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
            ("sensitive_onehot", OneHotEncoder(categories=[[privileged_value, protected_value]],handle_unknown='ignore'), sensitive_features),
            ("categorical_target", OneHotEncoder(categories=[[negative_class, positive_class]],handle_unknown='ignore'), target_class)
        ])

        train, test = train_test_split(dataframe, test_size=150, random_state=get_seed())
        train, validation = train_test_split(train, test_size=150, random_state=get_seed())

        train_sensitive = train[sensitive_features]
        validation_sensitive = validation[sensitive_features]
        test_sensitive = test[sensitive_features]

        ct.fit(train)
        train = ct.transform(train)
        validation = ct.transform(validation)
        test = ct.transform(test)

        num_classes = int(dataframe[target_class].nunique())
        num_features = train.shape[1] - num_classes
        num_sensitive = int(dataframe[sensitive_features].nunique())

        if self.sensitive_labels:
            num_target = num_classes + num_sensitive
        else:
            num_target = num_classes

        dataset = {
            'num_classes': num_classes,
            'num_features': num_features,
            'num_sensitive': num_sensitive,
            'train': {
                'features': train[:, :-num_classes],
                'labels': train[:, -num_target:],
                'sensitive': train_sensitive
            },
            'validation': {
                'features': validation[:, :-num_classes],
                'labels': validation[:, -num_target:],
                'sensitive': validation_sensitive
            },
            'test': {
                'features': test[:, :-num_classes],
                'labels': test[:, -num_target:],
                'sensitive': test_sensitive
            }
        }

        return dataset

    def _load_germancredit_age(self):
        path = os.path.join(BASE_PATH, 'german_credit', 'german.data')
        column_names = ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employmentsince',
                        'installmentate', 'personalstatus', 'garantors', 'residencesince', 'property', 'age',
                        'othereinstallments', 'housing', 'existingcredits', 'job', 'numbermaintence', 'telephone',
                        'foreign', 'label']

        dataframe = pd.read_csv(path, names=column_names, delim_whitespace=True)
        categorical_features = ['status', 'history', 'purpose', 'savings', 'employmentsince', 'personalstatus',
                                'garantors', 'property', 'othereinstallments', 'housing', 'job', 'telephone']
        numerical_features = ['duration', 'amount', 'installmentate', 'residencesince', 'existingcredits',
                              'numbermaintence']

        dataframe['label'] = dataframe['label'].astype(str)
        dataframe.loc[dataframe['label'] == '1','label'] = 'Good'
        dataframe.loc[dataframe['label'] == '2','label'] = 'Bad'

        dataframe['age_discrete'] = dataframe['age'].astype(str)
        dataframe.loc[dataframe['age'] <= 25, 'age_discrete'] = 'Younger'
        dataframe.loc[dataframe['age'] > 25,  'age_discrete'] = 'Older'

        sensitive_features = ['age_discrete']
        privileged_value = 'Older'
        protected_value = 'Younger'

        target_class = ['label']
        negative_class = 'Bad'
        positive_class = 'Good'

        ct = ColumnTransformer([
            ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
            ("sensitive_onehot", OneHotEncoder(categories=[[privileged_value, protected_value]],handle_unknown='ignore'), sensitive_features),
            ("categorical_target", OneHotEncoder(categories=[[negative_class, positive_class]],handle_unknown='ignore'), target_class)
        ])

        train, test = train_test_split(dataframe, test_size=150, random_state=get_seed())
        train, validation = train_test_split(train, test_size=150, random_state=get_seed())

        train_sensitive = train[sensitive_features]
        validation_sensitive = validation[sensitive_features]
        test_sensitive = test[sensitive_features]

        ct.fit(train)
        train = ct.transform(train)
        validation = ct.transform(validation)
        test = ct.transform(test)

        num_classes = int(dataframe[target_class].nunique())
        num_features = train.shape[1] - num_classes
        num_sensitive = int(dataframe[sensitive_features].nunique())

        if self.sensitive_labels:
            num_target = num_classes + num_sensitive
        else:
            num_target = num_classes

        dataset = {
            'num_classes': num_classes,
            'num_features': num_features,
            'num_sensitive': num_sensitive,
            'train': {
                'features': train[:, :-num_classes],
                'labels': train[:, -num_target:],
                'sensitive': train_sensitive
            },
            'validation': {
                'features': validation[:, :-num_classes],
                'labels': validation[:, -num_target:],
                'sensitive': validation_sensitive
            },
            'test': {
                'features': test[:, :-num_classes],
                'labels': test[:, -num_target:],
                'sensitive': test_sensitive
            }
        }

        return dataset


    def _load_income(self):
        train_path = os.path.join(BASE_PATH, 'income', 'adult.data')
        test_path = os.path.join(BASE_PATH, 'income', 'adult.test')

        column_names = ['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain',
                        'capital-loss', 'hours-per-week', 'native-country', 'label']

        categorical_features = ['workclass', 'education', 'marital-status',
                                'occupation', 'relationship', 'race',
                                'native-country']
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                              'capital-loss', 'hours-per-week']

        sensitive_features = ['sex']
        privileged_value = 'Male'
        protected_value = 'Female'

        target_class = ['label']
        negative_class = '<=50K'
        positive_class = '>50K'

        train_dataframe = pd.read_csv(train_path, names=column_names, sep=', ', engine='python')
        test = pd.read_csv(test_path, names=column_names, sep=', ', engine='python')
        test['label'] = test['label'].str.replace('.', '', regex=False)

        ct = ColumnTransformer([
            ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
            ("sensitive_onehot", OneHotEncoder(categories=[[privileged_value, protected_value]], handle_unknown='ignore'), sensitive_features),
            ("categorical_target", OneHotEncoder(categories=[[negative_class, positive_class]], handle_unknown='ignore'), target_class)
        ])

        train, validation = train_test_split(train_dataframe, test_size=0.2, random_state=get_seed())

        train_sensitive = train[sensitive_features]
        validation_sensitive = validation[sensitive_features]
        test_sensitive = test[sensitive_features]

        ct.fit(train)
        train = ct.transform(train).todense()
        validation = ct.transform(validation).todense()
        test = ct.transform(test).todense()

        num_classes = int(train_dataframe[target_class].nunique())
        num_features = train.shape[1] - num_classes
        num_sensitive = int(train_dataframe[sensitive_features].nunique())

        if self.sensitive_labels:
            num_target = num_classes + num_sensitive
        else:
            num_target = num_classes

        dataset = {
            'num_classes': num_classes,
            'num_features': num_features,
            'num_sensitive': num_sensitive,
            'train': {
                'features': train[:, :-num_classes],
                'labels': train[:, -num_target:],
                'sensitive': train_sensitive
            },
            'validation': {
                'features': validation[:, :-num_classes],
                'labels': validation[:, -num_target:],
                'sensitive': validation_sensitive
            },
            'test': {
                'features': test[:, :-num_classes],
                'labels': test[:, -num_target:],
                'sensitive': test_sensitive
            }
        }

        return dataset
