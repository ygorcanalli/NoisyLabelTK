# %%
import tensorflow as tf
import noisylabeltk.datasets as datasets
import noisylabeltk.models as models
import neptune
import datetime
import numpy as np
from neptunecontrib.monitoring.keras import NeptuneMonitor

#%%
PARAMS = {
    'batch_size': 64,
    'epochs': 10,
    'dataset': 'breast-cancer',
    'model': 'simple_mlp'}

neptune.init(project_qualified_name='ygorcanalli/sandbox')
exp = neptune.create_experiment(name="noiselabeltk-sandbox",
                            description="lidsa",
                            #tags=["breast-cancer", "simple_mlp"],
                            params=PARAMS)

log_dir = "tflogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# %%
(train_ds, validation_ds, test_ds), num_features, num_classes = datasets.load_dataset(PARAMS['dataset'])

exp.set_property('num_features', num_features)
exp.set_property('num_classes', num_classes)

model = models.create_model(PARAMS['model'], num_features,  num_classes)
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.summary(print_fn=lambda x: neptune.log_text('model_summary', x))
#%%
history = model.fit(train_ds, epochs=10, 
                    validation_data=validation_ds,
                    callbacks=[NeptuneMonitor()])
eval_metrics = model.evaluate(test_ds, verbose=0)

for j, metric in enumerate(eval_metrics):
    neptune.log_metric('eval_' + model.metrics_names[j], metric)
