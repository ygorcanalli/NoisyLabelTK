# %%
import tensorflow as tf
import datasets as datasets
import models as models
import neptune
import datetime
from neptunecontrib.monitoring.keras import NeptuneMonitor

#%%
PARAMS = {'lr': 0.005,
          'momentum': 0.9,
          'batch_size': 64,
          'epochs': 10}

neptune.init(project_qualified_name='ygorcanalli/sandbox')
exp = neptune.create_experiment(name="noiselabeltk-sandbox",
                            description="lidsa",
                            tags=["fashionmnist", "simple_conv"],
                            params=PARAMS)

log_dir = "tflogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# %%
(train_ds, validation_ds, test_ds), ds_info = datasets.load_germancredit()

exp.set_property('ds_info', ds_info)
model = models.simple_mlp_32_64(ds_info)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))

history = model.fit(train_ds, epochs=10, 
                    validation_data=validation_ds,
                    callbacks=[NeptuneMonitor()])
eval_metrics = model.evaluate(test_ds, verbose=0)
for j, metric in enumerate(eval_metrics):
    neptune.log_metric('eval_' + model.metrics_names[j], metric)
