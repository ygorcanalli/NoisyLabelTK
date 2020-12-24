# %%
import tensorflow as tf
import datasets
import models
# %%
(train_ds, validation_ds, test_ds), ds_info = datasets.load_fashionmnist()
model = models.simple_conv_32_64(ds_info)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, 
                    validation_data=validation_ds)
score = model.evaluate(test_ds, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# %%

(train_ds, validation_ds, test_ds), ds_info = datasets.load_cifar10()
model = models.vgg(ds_info)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, 
                    validation_data=validation_ds)
score = model.evaluate(test_ds, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
