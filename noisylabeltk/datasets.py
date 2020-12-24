#%%
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#%%
def load_mnist():
    (train_ds, validation_ds, test_ds), ds_info = tfds.load('mnist',
                                    split=['train[0:90%]', 'train[-10%:]','test'],
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

def load_fashionmnist():
    (train_ds, validation_ds, test_ds), ds_info = tfds.load('fashion_mnist',
                                    split=['train[0:90%]', 'train[-10%:]','test'],
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

def load_cifar10():
    (train_ds, validation_ds, test_ds), ds_info = tfds.load('cifar10', 
                                    split=['train[0:90%]', 'train[-10%:]','test'],
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

def load_cifar10():
    (train_ds, validation_ds, test_ds), ds_info = tfds.load('cifar100', 
                                    split=['train[0:90%]', 'train[-10%:]','test'],
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


def load_food101():
    (train_ds, validation_ds, test_ds), ds_info = tfds.load('food101', 
                                    split=['train[0:90%]', 'train[-10%:]','test'],
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
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label
# %%
