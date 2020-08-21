import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import os
import cProfile
import pstats
import random
# from pstats import SortKey
import timeit
import pathlib
import re
import io
import glob
import PIL
import PIL.Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %%
path = '/Users/Max/OneDrive - Imperial College London/4th yr project/GAN-Challenge/scenes'
# iters = int(sys.argv[1])

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
data_dir = pathlib.Path('scenes/spirited_away/')
images = list(data_dir.glob('*.jpeg'))

batch_size = 32
img_height = 28
img_width = 28
channels=3
buffer_size=200

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width),label_mode=None, batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width),  label_mode=None, batch_size=32)

train_ds.element_spec

train_ds = train_ds.shuffle(buffer_size)
# train_ds = train_ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
# train_labels = tf.data.Dataset.from_tensor_slices(np.float32(np.ones(train_ds.cardinality())))
# train=tf.data.Dataset.zip((train_ds,train_labels))
# val_labels = tf.data.Dataset.from_tensor_slices(np.float32(np.ones(val_ds.cardinality())))
# val=tf.data.Dataset.zip((val_ds,val_labels))
# %%
dropout = 0.4

discriminator_input = keras.Input(shape=(img_height, img_width, channels))
x = layers.Conv2D(16, 3, strides=2, activation='relu',
                  padding='same')(discriminator_input)
x = layers.Dropout(dropout)(x)
x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = layers.Dropout(dropout)(x)
x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = layers.Dropout(dropout)(x)
x = layers.Flatten()(x)
x = layers.Dense(1)(x)
discriminator_output = layers.Activation('sigmoid')(x)
discriminator = keras.Model(
    discriminator_input, discriminator_output, name='discriminator')
discriminator.summary()
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])

# %%
dropout = 0.4
depth = 64+64+64+64
dim = np.int32(img_height/4)
latent_dim = 100
# In: 100
# Out: dim x dim x depth
generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(dim*dim*depth, input_dim=100)(generator_input)
x = layers.BatchNormalization(momentum=0.6)(x)
x = layers.Activation('relu')(x)
x = layers.Reshape((dim, dim, depth))(x)
x = layers.Dropout(dropout)(x)
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
x = layers.UpSampling2D()(x)
x = layers.Conv2DTranspose(int(depth/2), 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.6)(x)
x = layers.Activation('relu')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2DTranspose(int(depth/4), 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.6)(x)
x = layers.Activation('relu')(x)
x = layers.Conv2DTranspose(int(depth/8), 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.6)(x)
x = layers.Activation('relu')(x)
# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
x = layers.Conv2DTranspose(channels, 5, padding='same')(x)
generator_output = layers.Activation('sigmoid')(x)
generator = keras.Model(generator_input, generator_output, name='generator')
generator.summary()

# %%
batch_size = 32

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1))*random.random()*0.7+0.5, tf.zeros((batch_size, 1))*random.random()*0.3-0.4], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

# %%
# Prepare the dataset. We use both the training & test MNIST digits.
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

gan.fit(train_ds, epochs=100)

# %%
'''
pr = cProfile.Profile()
pr.enable()

for i in range(1):
    train_on_n()

pr.disable()
pr.print_stats('cumulative')
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)
'''
# %%
rand = np.random.rand(1000, 100)
# GAN.train_on_batch(rand,[1])
img = generator.predict(np.random.rand(1, 100))
img.shape
img = img.reshape(img_width,img_height,channels)
img
plt.imshow(img)
plt.imsave("fig.png", img, dpi=300)

discriminator.save("discriminator_model.hdf5")
generator.save("generator_model.hdf5")
