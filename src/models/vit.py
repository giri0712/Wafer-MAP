"""
vit.py — Lightweight Vision Transformer for wafer maps
======================================================================
~400K-param ViT-Tiny variant tuned for 52x52 single-channel wafer maps:
  - 8x8 patches -> 42 patch tokens
  - 6 transformer encoder blocks, 4 heads, model dim 96, MLP ratio 2
  - MultiHeadAttention layers are NAMED (block_{i}_mha) so the
    attention-rollout explainability code can introspect them.

Design notes:
  - No [CLS] token: mean-pooling over patch tokens is used instead,
    which keeps the parameter budget low and works well on small maps.
  - Deliberately tiny: the research claim is that a *lightweight*
    transformer with diffusion augmentation rivals a much larger CNN
    while remaining deployable on fab-edge hardware.
"""

import tensorflow as tf
from tensorflow.keras import layers

NUM_CLASSES = 9
IMG_SIZE = 52
PATCH = 8
DIM = 96
DEPTH = 6
HEADS = 4
MLP_RATIO = 2
DROP = 0.1


@tf.keras.utils.register_keras_serializable(package="wafer")
class PatchExtract(layers.Layer):
    """(B, H, W, 1) -> (B, num_patches, dim) non-overlapping patch embedding."""

    def __init__(self, patch=PATCH, dim=DIM, **kw):
        super().__init__(**kw)
        self.patch = patch
        self.dim = dim
        self.proj = layers.Dense(dim, name="patch_proj")

    def build(self, input_shape):
        patch_dim = self.patch * self.patch * int(input_shape[-1])
        self.proj.build((None, patch_dim))
        super().build(input_shape)

    def call(self, x):
        s = tf.shape(x)
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch, self.patch, 1],
            strides=[1, self.patch, self.patch, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )                                    # (B, gh, gw, patch*patch*C)
        patches = tf.reshape(patches, (s[0], -1, int(patches.shape[-1])))
        return self.proj(patches)            # (B, N, dim)

    def compute_output_shape(self, input_shape):
        b = input_shape[0]
        n = (input_shape[1] // self.patch) * (input_shape[2] // self.patch)
        return (b, n, self.dim)

    def get_config(self):
        cfg = super().get_config()
        cfg["patch"] = self.patch
        cfg["dim"] = self.dim
        return cfg


def transformer_encoder(x, index: int):
    """One pre-norm transformer encoder block with named attention layer."""
    h = layers.MultiHeadAttention(
        num_heads=HEADS, key_dim=DIM // HEADS, dropout=DROP,
        name=f"block_{index}_mha")(x, x)
    x = layers.Add(name=f"block_{index}_res1")([x, h])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"block_{index}_ln1")(x)

    y = layers.Dense(DIM * MLP_RATIO, activation="gelu",
                     name=f"block_{index}_fc1")(x)
    y = layers.Dense(DIM, name=f"block_{index}_fc2")(y)
    x = layers.Add(name=f"block_{index}_res2")([x, y])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"block_{index}_ln2")(x)
    return x


def build_lightweight_vit(num_classes: int = NUM_CLASSES,
                          img_size: int = IMG_SIZE) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 1), name="wafer_input")

    x = PatchExtract(name="patch_extract")(inputs)          # (B, 42, 128)
    x = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(x)
    x = layers.Dropout(DROP)(x)

    for i in range(DEPTH):
        x = transformer_encoder(x, i)

    x = layers.GlobalAveragePooling1D(name="token_pool")(x)  # mean over tokens
    x = layers.Dropout(DROP)(x)
    x = layers.Dense(128, activation="gelu", name="head_dense")(x)
    x = layers.Dropout(DROP)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="WaferViT")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3):
    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000, alpha=0.05)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    return model


if __name__ == "__main__":
    m = build_lightweight_vit()
    compile_model(m)
    m.summary()
    print(f"\nTotal params: {m.count_params():,}")
