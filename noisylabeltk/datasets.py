#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import os
import sklearn.datasets as skds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from noisylabeltk.noise_generator import PairwiseLabelNoiseGenerator

BASE_PATH = "datasets"

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


#%%

class DatasetLoader(object):

    def __init__(self, name):
        self.name =  name
        self.datasets = {
            'german': self.load_germancredit,
            'breast-cancer': self.load_breastcancer
        }

    def load(self):
        dataset_loader = self.datasets[self.name]
        dataset = dataset_loader()
        train_ds = np_to_dataset(dataset['train']['features'], dataset['train']['labels'])
        validation_ds = np_to_dataset(dataset['validation']['features'], dataset['validation']['labels'])
        test_ds = np_to_dataset(dataset['test']['features'], dataset['test']['labels'])

        return (train_ds, validation_ds, test_ds), dataset['num_features'], dataset['num_classes']

    def pollute_and_load(self, T):
        dataset_loader = self.datasets[self.name]
        dataset = dataset_loader()

        noise_generator = PairwiseLabelNoiseGenerator(dataset['train']['labels'], T)
        noisy_train_labels = noise_generator.generate_noisy_labels()

        train_ds = np_to_dataset(dataset['train']['features'], noisy_train_labels)
        validation_ds = np_to_dataset(dataset['validation']['features'], dataset['validation']['labels'])
        test_ds = np_to_dataset(dataset['test']['features'], dataset['test']['labels'])

        return (train_ds, validation_ds, test_ds), dataset['num_features'], dataset['num_classes']

    def load_mnist(self):
        (train_ds, validation_ds, test_ds), ds_info = tfds.load('mnist',
                                                                split=['train[0:90%]', 'train[-10%:]', 'test'],
                                                                as_supervised=True,
                                                                with_info=True)

        train_ds = train_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train[0:90%]'].num_examples)
        train_ds = train_ds.batch(256)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.shuffle(ds_info.splits['train[-10%:]'].num_examples)
        validation_ds = validation_ds.batch(256)
        validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(256)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, validation_ds, test_ds), ds_info
    def load_fashionmnist(self):
        (train_ds, validation_ds, test_ds), ds_info = tfds.load('fashion_mnist',
                                                                split=['train[0:90%]', 'train[-10%:]', 'test'],
                                                                as_supervised=True,
                                                                with_info=True)

        train_ds = train_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train[0:90%]'].num_examples)
        train_ds = train_ds.batch(256)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.shuffle(ds_info.splits['train[-10%:]'].num_examples)
        validation_ds = validation_ds.batch(256)
        validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(256)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, validation_ds, test_ds), ds_info
    def load_cifar10(self):
        (train_ds, validation_ds, test_ds), ds_info = tfds.load('cifar10',
                                                                split=['train[0:90%]', 'train[-10%:]', 'test'],
                                                                shuffle_files=True,
                                                                as_supervised=True,
                                                                with_info=True)

        train_ds = train_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train[0:90%]'].num_examples)
        train_ds = train_ds.batch(128)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.shuffle(ds_info.splits['train[-10%:]'].num_examples)
        validation_ds = validation_ds.batch(128)
        validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(128)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, validation_ds, test_ds), ds_info
    def load_cifar100(self):
        (train_ds, validation_ds, test_ds), ds_info = tfds.load('cifar100',
                                                                split=['train[0:90%]', 'train[-10%:]', 'test'],
                                                                shuffle_files=True,
                                                                as_supervised=True,
                                                                with_info=True)

        train_ds = train_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train[0:90%]'].num_examples)
        train_ds = train_ds.batch(256)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.shuffle(ds_info.splits['train[-10%:]'].num_examples)
        validation_ds = validation_ds.batch(256)
        validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(256)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, validation_ds, test_ds), ds_info
    def load_food101(self):
        (train_ds, validation_ds, test_ds), ds_info = tfds.load('food101',
                                                                split=['train[0:90%]', 'train[-10%:]', 'test'],
                                                                shuffle_files=True,
                                                                as_supervised=True,
                                                                with_info=True)

        train_ds = train_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(ds_info.splits['train[0:90%]'].num_examples)
        train_ds = train_ds.batch(256)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        validation_ds = validation_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_ds = validation_ds.cache()
        validation_ds = validation_ds.shuffle(ds_info.splits['train[-10%:]'].num_examples)
        validation_ds = validation_ds.batch(256)
        validation_ds = validation_ds.prefetch(tf.data.experimental.AUTOTUNE)

        test_ds = test_ds.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(256)
        test_ds = test_ds.cache()
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, validation_ds, test_ds), ds_info
    def load_svhn(self):
        (train_ds, test_ds), ds_info = tfds.load('svhn_cropped',
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)
        return (train_ds, test_ds), ds_info
    def load_imagenet(self):
        (train_ds, test_ds), ds_info = tfds.load('imagenet2012',
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)
        return (train_ds, test_ds), ds_info
    def load_openimage(self):
        (train_ds, test_ds), ds_info = tfds.load('open_images_challenge2019_detection',
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)
        return (train_ds, test_ds), ds_info
    # TODO: migrate german loader to numpy
    def load_germancredit(self):
        path = os.path.join(BASE_PATH, 'german_credit', 'german.data')
        column_names = ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employmentsince',
                        'installmentate', 'personalstatus', 'garantors', 'residencesince', 'property', 'age',
                        'othereinstallments', 'housing', 'existingcredits', 'job', 'numbermaintence', 'telephone',
                        'foreign', 'label']

        dataframe = pd.read_csv(path, names=column_names, delim_whitespace=True)
        categorical_features = ['status', 'history', 'purpose', 'savings', 'employmentsince', 'personalstatus',
                                'garantors', 'property', 'othereinstallments', 'housing', 'job', 'telephone', 'foreign']
        numerical_features = ['duration', 'amount', 'installmentate', 'residencesince', 'age', 'existingcredits',
                              'numbermaintence']
        target_class = 'label'

        # categorical_features = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        # numerical_features = [1, 4, 7, 10, 12, 15, 17]
        # target_class = [20]
        ct = ColumnTransformer([
            ("categorical_onehot", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
            ("categorical_target", OneHotEncoder(handle_unknown='ignore'), target_class),
        ])

        train, test = train_test_split(dataframe, test_size=0.2)
        train, validation = train_test_split(train, test_size=0.2)

        ct.fit(train)
        train = ct.transform(train)
        validation = ct.transform(validation)
        test = ct.transform(test)

        num_classes = len(dataframe[target_class].unique())
        num_features = train.shape[1] - num_classes

        train_ds = np_to_dataset(train, num_classes)
        validation_ds = np_to_dataset(validation, num_classes)
        test_ds = np_to_dataset(test, num_classes)

        return (train_ds, validation_ds, test_ds), num_features, num_classes
    def load_breastcancer(self):
        breast_cancer = skds.load_breast_cancer(as_frame=True)
        dataframe = breast_cancer['frame']

        numerical_features = breast_cancer['feature_names']

        target_class = ['target']
        ct = ColumnTransformer([
            ("numerical", StandardScaler(), numerical_features),
            ("categorical_target", OneHotEncoder(handle_unknown='ignore'), target_class),
        ])

        train, test = train_test_split(dataframe, test_size=0.2)
        train, validation = train_test_split(train, test_size=0.2)

        ct.fit(train)
        train = ct.transform(train)
        validation = ct.transform(validation)
        test = ct.transform(test)


        num_classes = len(breast_cancer['target_names'])
        num_features = train.shape[1] - num_classes

        dataset = {
            'num_classes': num_classes,
            'num_features': num_features,
            'train': {
                'features': train[:, :-num_classes],
                'labels': train[:, -num_classes:]
            },
            'validation': {
                'features': validation[:, :-num_classes],
                'labels': validation[:, -num_classes:]
            },
            'test': {
                'features': test[:, :-num_classes],
                'labels': test[:, -num_classes:]
            }
        }

        return dataset

