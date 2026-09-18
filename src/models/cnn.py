"""
models/cnn.py — CNN baseline (ablation arm 1)
======================================================================
The original 6-block WaferMapCNN, kept as the ablation baseline so the
ViT claims are benchmarked against a strong, published-style CNN.
Input (52, 52, 1). Layer names preserved for Grad-CAM ("conv6").
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

NUM_CLASSES = 9
IMG_SIZE = 52


def build_cnn(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 1), name="wafer_input")

    def block(x, filters, name):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(1e-4),
                          name=f"{name}a")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                          name=f"{name}b")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        return x

    x = block(inputs, 32, "conv1_")
    x = block(x, 64, "conv3_")
    x = block(x, 128, "conv5_")

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = models.Model(inputs, outputs, name="WaferMapCNN")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
    print(f"\nTotal params: {m.count_params():,}")
