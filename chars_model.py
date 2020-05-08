import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

def CModel(size, class_num, weights):
    resnet = tf.keras.applications.ResNet101V2(classes=class_num, include_top=False, weights=weights, input_shape=(size, size, 3))

    output = resnet.layers[-1].output
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)

    base_model = tf.keras.models.Model(inputs=resnet.input, outputs=output)
    if not weights is None:
        for layer in base_model.layers:
            layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(class_num, activation='softmax'))
    return model