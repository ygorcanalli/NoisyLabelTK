#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
def load_mnist():
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
    return train_ds, test_ds

def load_fashionmnist():
    train_ds, test_ds = tfds.load('fashion_mnist', split=['train', 'test'])
    return train_ds, test_ds

def load_cifar10():
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'])
    return train_ds, test_ds

def load_cifar100():
    train_ds, test_ds = tfds.load('cifar100', split=['train', 'test'])
    return train_ds, test_ds

def load_svhn():
    train_ds, test_ds = tfds.load('svhn_cropped', split=['train', 'test'])
    return train_ds, test_ds

def load_imagenet():
    train_ds, test_ds = tfds.load('imagenet2012', split=['train', 'test'])
    return train_ds, test_ds

def load_openimage():
    train_ds, test_ds = tfds.load('open_images_challenge2019_detection', split=['train', 'test'])
    return train_ds, test_ds

def load_germancredit():
    train_ds, test_ds = tfds.load('german_credit_numeric', split=['train[:80%]', 'train[-20%:]'])
    return train_ds, test_ds

    

# %%
