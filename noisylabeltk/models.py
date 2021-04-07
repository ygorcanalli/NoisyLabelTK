from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization

def simple_mlp(input_shape, num_classes, *args, **kwargs):
    num_layers = kwargs['num-layers']
    dropout_rate = kwargs['dropout']
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    for i in range(num_layers):
        model.add(Dense(kwargs['hidden-size'], activation='relu'))

    model.add(Dense(num_classes, activation="softmax"))
    return model

# Taken from: https://keras.io/examples/vision/mnist_convnet/
def simple_conv_32_64(input_shape, num_classes):
    model = Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5, seed=get_seed()),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model

# Taken from: https://github.com/geifmany/cifar-vgg
def vgg(ds_info):
    num_classes = ds_info.features["label"].num_classes
    input_shape = ds_info.features["image"].shape

    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=input_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3, seed=get_seed()))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4, seed=get_seed()))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5, seed=get_seed()))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5, seed=get_seed()))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


models = {
    'simple-mlp': simple_mlp,
    'simple-conv': simple_conv_32_64
}

def create_model(model_name, input_shape, num_classes, *args, **kwargs):
    model_function = models[model_name]
    return model_function(input_shape, num_classes, *args, **kwargs)

