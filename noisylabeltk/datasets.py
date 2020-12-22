#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
def load_mnist():
    (train_ds, test_ds), ds_info = tfds.load('mnist',
                                    split=['train', 'test'],
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

def load_fashionmnist():
    (train_ds, test_ds), ds_info = tfds.load('fashion_mnist',
                                    split=['train', 'test'],
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

def load_cifar10():
    (train_ds, test_ds), ds_info = tfds.load('cifar10', 
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True)
    

    return (train_ds, test_ds), ds_info

def load_cifar100():
    (train_ds, test_ds), ds_info = tfds.load('cifar100', 
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True)
    

    return (train_ds, test_ds), ds_info

def load_svhn():
    (train_ds, test_ds), ds_info = tfds.load('svhn_cropped',
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

def load_imagenet():
    (train_ds, test_ds), ds_info = tfds.load('imagenet2012',
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

def load_openimage():
    (train_ds, test_ds), ds_info = tfds.load('open_images_challenge2019_detection',
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

def load_germancredit():
    (train_ds, test_ds), ds_info = tfds.load('german_credit_numeric',
                                    split=['train[:80%]', 'train[-20%:]'],
                                    as_supervised=True,
                                    with_info=True)
    return (train_ds, test_ds), ds_info

    

# %%
