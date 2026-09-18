"""
diffusion/latent_diffusion.py — SD-style latent diffusion, from scratch
======================================================================
Stable-Diffusion *architecture* at toy scale, trained only on wafer maps
(no pretrained checkpoints, so it stays a self-contained contribution):

  1. VAE      52x52x1 wafer map -> 13x13x4 latent   (4x spatial squeeze)
  2. UNet     eps-prediction diffusion in latent space,
              class-conditional via embeddings + classifier-free guidance
  3. Sampler  DDIM-style deterministic sampler for fast generation

Why latent space? Pixel-space DDPMs need hundreds of steps to look good;
the VAE removes wafer-map redundancy (large flat background) so the UNet
converges in a fraction of the steps — realistic for a course budget.

Usage:
    model = LatentDiffusionModel(num_classes=9)
    model.train_vae(X_train, epochs=30)      # via model.vae.fit(...)
    model.train_unet(X_train, y_train)       # via model.fit(ds)
    model.save(ckpt_dir); model.load(ckpt_dir)
    samples = model.sample(class_label="Scratch", n=16)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMG_SIZE = 52
LATENT_SIZE = 13          # 52 / 4
LATENT_CH = 4
TIMESTEPS = 400           # diffusion training steps
DDIM_STEPS = 50           # default sampling steps
NUM_CLASSES = 9
GUIDANCE_SCALE = 2.5      # classifier-free guidance weight at sampling

CLASS_ORDER = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]


def cosine_alphas_bar(timesteps: int = TIMESTEPS, s: float = 0.008):
    """Nichol & Dhariwal cosine noise schedule (better for small data)."""
    t = np.linspace(0, timesteps, timesteps + 1)
    f = np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_bar = f / f[0]
    return alphas_bar[1:].astype(np.float32)          # (T,)


class DiffusionSchedule:
    """Precomputed alpha/alpha_bar tensors for training and sampling."""

    def __init__(self, timesteps: int = TIMESTEPS):
        ab = cosine_alphas_bar(timesteps)
        self.T = timesteps
        self.alphas_bar = tf.constant(ab)                       # a_bar_t

    def gather(self, tensor, t):
        return tf.gather(tensor, t)

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
        ab = tf.reshape(self.gather(self.alphas_bar, t), (-1, 1, 1, 1))
        return tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * noise


# ----------------------------------------------------------------------
# Custom serializable layers
# ----------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="wafer")
class TimeEmbedding(layers.Layer):
    """Sinusoidal timestep embedding (wrapped so it works in Keras 3)."""

    def __init__(self, dim=128, **kw):
        super().__init__(**kw)
        self.dim = dim

    def call(self, t):
        half = self.dim // 2
        freqs = tf.exp(-tf.math.log(10000.0) *
                       tf.range(half, dtype=tf.float32) / half)
        ang = tf.cast(t, tf.float32)[:, None] * freqs[None, :] * 1000.0
        return tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)

    def get_config(self):
        cfg = super().get_config()
        cfg["dim"] = self.dim
        return cfg


@tf.keras.utils.register_keras_serializable(package="wafer")
class BroadcastAdd(layers.Layer):
    """(B,H,W,C) feature map + (B,C) embedding -> broadcast add (FiLM-style)."""

    def call(self, inputs):
        x, e = inputs
        e = tf.reshape(e, (-1, 1, 1, tf.shape(e)[-1]))
        return x + e

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# ----------------------------------------------------------------------
# VAE  (wafer maps are near-binary; reconstruction is easy, so this
#       stays tiny — the latent simply strips background redundancy)
# ----------------------------------------------------------------------
def build_vae() -> tf.keras.Model:
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    e = layers.Conv2D(32, 3, 2, "same", activation="silu")(inp)   # 26
    e = layers.Conv2D(64, 3, 2, "same", activation="silu")(e)     # 13
    e = layers.Conv2D(64, 3, padding="same", activation="silu")(e)
    z = layers.Conv2D(LATENT_CH, 3, padding="same", name="z")(e)  # 13x13x4

    d = layers.Conv2D(64, 3, padding="same", activation="silu")(z)
    d = layers.UpSampling2D(2)(d)
    d = layers.Conv2D(32, 3, padding="same", activation="silu")(d)
    d = layers.UpSampling2D(2)(d)
    d = layers.Conv2D(16, 3, padding="same", activation="silu")(d)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid",
                        name="recon")(d)
    return tf.keras.Model(inp, out, name="wafer_vae")


class VAE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = build_vae()
        # encoder: image -> latent ; decoder: latent -> reconstruction
        self.encoder = tf.keras.Model(self.net.input,
                                      self.net.get_layer("z").output)
        self.decoder = tf.keras.Model(self.net.get_layer("z").output,
                                      self.net.get_layer("recon").output)

    def compile(self, lr=2e-4):
        super().compile()
        self.opt = tf.keras.optimizers.Adam(lr)
        self.recon_loss = tf.keras.losses.MeanSquaredError()

    def train_step(self, batch):
        x = batch if isinstance(batch, tf.Tensor) else batch[0]
        with tf.GradientTape() as tape:
            x_rec = self.net(x, training=True)
            loss = self.recon_loss(x, x_rec)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
        return {"recon": loss}

    def call(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Class-conditional UNet (eps-predictor) in latent space
# ----------------------------------------------------------------------
def resblock(x, filters, emb, name):
    h = layers.Conv2D(filters, 3, padding="same", activation="silu",
                      name=f"{name}_c1")(x)
    e = layers.Dense(filters, name=f"{name}_emb")(emb)   # (B, filters)
    h = BroadcastAdd(name=f"{name}_badd")([h, e])
    h = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c2")(h)
    r = layers.Conv2D(filters, 1, name=f"{name}_skip")(x) \
        if x.shape[-1] != filters else x
    return layers.Add()([r, h])


def build_unet(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    z_in = layers.Input((LATENT_SIZE, LATENT_SIZE, LATENT_CH))
    t_in = layers.Input(())
    c_in = layers.Input(())          # int32 class index (NULL = num_classes)

    temb = layers.Dense(128, activation="silu")(TimeEmbedding(128)(t_in))
    # +1 row: index `num_classes` is the learned NULL class used for
    # classifier-free guidance (label dropout during training)
    cemb = layers.Embedding(num_classes + 1, 128)(c_in)
    emb = layers.Concatenate()([temb, cemb])                 # (B, 256)
    emb = layers.Dense(128, activation="silu")(emb)

    x = layers.Conv2D(64, 3, padding="same")(z_in)
    x = resblock(x, 64, emb, "res1")
    x = resblock(x, 64, emb, "res2")

    d1 = resblock(x, 128, emb, "down1")
    x = layers.AveragePooling2D(pool_size=2)(d1)             # 6x6x128
    x = resblock(x, 128, emb, "mid1")

    x = layers.Resizing(LATENT_SIZE, LATENT_SIZE)(x)         # back to 13x13
    x = resblock(layers.Concatenate()([x, d1]), 64, emb, "up1")
    x = resblock(x, 64, emb, "up2")

    out = layers.Conv2D(LATENT_CH, 3, padding="same", name="eps_out")(x)
    return tf.keras.Model([z_in, t_in, c_in], out, name="wafer_unet")


# ----------------------------------------------------------------------
# LatentDiffusionModel — trainer + DDIM sampler
# ----------------------------------------------------------------------
class LatentDiffusionModel(tf.keras.Model):
    """Holds VAE + UNet; trains UNet with eps-prediction loss."""

    def __init__(self, num_classes: int = NUM_CLASSES, timesteps=TIMESTEPS):
        super().__init__()
        self.num_classes = num_classes
        self.T = timesteps
        self.sched = DiffusionSchedule(timesteps)
        self.vae = VAE()
        self.unet = build_unet(num_classes)

    # ---------------- training ----------------
    def compile(self, lr=2e-4, lr_unet=None):
        super().compile()
        self.opt = tf.keras.optimizers.Adam(lr if lr_unet is None else lr_unet)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(self, batch):
        x, y = batch
        b = tf.shape(x)[0]
        y = tf.cast(y, tf.int32)
        # classifier-free guidance: replace label with NULL class 10% of time
        drop = tf.random.uniform((b,)) < 0.1
        y_in = tf.where(drop, tf.fill((b,), self.num_classes), y)
        # encode with frozen VAE encoder
        z0 = self.vae.encoder(x, training=False)
        t = tf.random.uniform((b,), 0, self.T, tf.int32)
        eps = tf.random.normal(tf.shape(z0))
        zt = self.sched.q_sample(z0, t, eps)
        with tf.GradientTape() as tape:
            eps_hat = self.unet([zt, t, y_in], training=True)
            loss = self.loss_fn(eps, eps_hat)
        grads = tape.gradient(loss, self.unet.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.unet.trainable_variables))
        return {"unet_loss": loss}

    def call(self, x):
        return self.vae.net(x)

    # ---------------- encoders / samplers ----------------
    @tf.function(reduce_retracing=True)
    def encode(self, x):
        return self.vae.encoder(x, training=False)

    @tf.function(reduce_retracing=True)
    def _ddim_step(self, zt, eps, t_idx, ab_prev):
        ab_t = tf.gather(self.sched.alphas_bar, t_idx)
        z0 = (zt - tf.sqrt(1.0 - ab_t) * eps) / tf.sqrt(ab_t)
        return tf.sqrt(ab_prev) * z0 + tf.sqrt(1.0 - ab_prev) * eps

    def sample(self, n: int, class_label=None, class_idx=None,
               steps: int = DDIM_STEPS, guidance: float = GUIDANCE_SCALE,
               seed: int = None):
        """DDIM sampling with classifier-free guidance."""
        if class_idx is None:
            class_idx = 0 if class_label is None else \
                CLASS_ORDER.index(class_label)
        ids = tf.fill((n,), tf.constant(int(class_idx), tf.int32))
        if seed is not None:
            zt = tf.random.stateless_normal(
                (n, LATENT_SIZE, LATENT_SIZE, LATENT_CH),
                seed=[seed, seed])
        else:
            zt = tf.random.normal((n, LATENT_SIZE, LATENT_SIZE, LATENT_CH))

        step_idx = np.linspace(self.T - 1, 0, steps, dtype=np.int64)
        null_ids = tf.fill((n,), tf.constant(self.num_classes, tf.int32))
        for i, t_cur in enumerate(step_idx):
            t = tf.fill((n,), tf.constant(int(t_cur), tf.int32))
            # CFG: conditional + unconditional (NULL class) branches
            eps_c = self.unet([zt, t, ids], training=False)
            eps_u = self.unet([zt, t, null_ids], training=False)
            eps = eps_u + guidance * (eps_c - eps_u)
            ab_prev = tf.constant(1.0, tf.float32) if i == steps - 1 else \
                tf.gather(self.sched.alphas_bar, step_idx[i + 1])
            zt = self._ddim_step(zt, eps, int(t_cur), ab_prev)

        # decode latent -> wafer map
        return self.vae.decoder(zt, training=False).numpy()

    # ---------------- persistence ----------------
    def save(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.vae.net.save(os.path.join(ckpt_dir, "vae.keras"))
        self.unet.save(os.path.join(ckpt_dir, "unet.keras"))
        with open(os.path.join(ckpt_dir, "meta.txt"), "w") as f:
            f.write(f"T={self.T}\nnum_classes={self.num_classes}\n")

    @classmethod
    def load(cls, ckpt_dir: str):
        meta = {}
        p = os.path.join(ckpt_dir, "meta.txt")
        if os.path.exists(p):
            for line in open(p):
                if "=" in line:
                    k, v = line.strip().split("=")
                    meta[k] = v
        m = cls(num_classes=int(meta.get("num_classes", NUM_CLASSES)),
                timesteps=int(meta.get("T", TIMESTEPS)))
        m.vae.net = tf.keras.models.load_model(
            os.path.join(ckpt_dir, "vae.keras"), compile=False)
        m.vae.encoder = tf.keras.Model(m.vae.net.input,
                                       m.vae.net.get_layer("z").output)
        m.vae.decoder = tf.keras.Model(m.vae.net.get_layer("z").output,
                                       m.vae.net.get_layer("recon").output)
        m.unet = tf.keras.models.load_model(
            os.path.join(ckpt_dir, "unet.keras"), compile=False)
        return m
