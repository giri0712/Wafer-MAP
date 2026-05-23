"""
model.py
CNN architecture for WM-811K wafer map defect pattern classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


NUM_CLASSES = 9
IMG_SIZE    = 64


def build_cnn(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> tf.keras.Model:
    """
    Build a CNN for wafer map defect pattern classification.
    Input:  (img_size, img_size, 1)
    Output: (num_classes,) softmax probabilities
    """
    inputs = layers.Input(shape=(img_size, img_size, 1), name="wafer_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv5")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv6")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="WaferMapCNN")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 0.001) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_cnn()
    compile_model(model)
    model.summary()
