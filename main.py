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
import timeit
import pathlib
import re
import io
import glob
import PIL
import PIL.Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# %%
iters = int(sys.argv[1])
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
data_dir = pathlib.Path('scenes/spirited_away/')
images = list(data_dir.glob('*.jpeg'))

batch_size = 32
img_height = 32
img_width = 32
channels=3
buffer_size=100

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width),label_mode=None, batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width),  label_mode=None, batch_size=32)
for element in train_ds.take(1).as_numpy_iterator():
    image = element[0]

plt.imshow(image/max(np.concatenate(np.concatenate(image))))
train_ds = train_ds.repeat(iters//train_ds.cardinality()+1).shuffle(buffer_size)

batch_size = 32
img_height = 28
img_width = 28
channels=3
buffer_size=200

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width),label_mode=None, batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'scenes', validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width),  label_mode=None, batch_size=32)
# %%
dropout = 0.4

discriminator_input = keras.Input(shape=(img_height, img_width, channels))
discriminator_input.shape
discriminator_input.dtype
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
dim = np.int(img_height/4)
# In: 100
# Out: dim x dim x depth
generator_input = keras.Input(shape=(100,))
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
discriminator.trainable = False
z = keras.Input(shape=(100,))
gen = generator(z)
dis = discriminator(gen)
GAN = keras.Model(z,dis)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer,
metrics=['accuracy'])
GAN.summary()

# %%
iterator = iter(train_ds)

for i in range(iters):
    # load real images (for later use)
    true_images = iterator.get_next()

    batch_size=len(true_images)

    # generate images
    noise_gen = np.random.rand(batch_size,100)
    generated_images = generator.predict(noise_gen)

    # get discriminator prediction
    disc_out = np.random.rand(batch_size,1)*0.5+0.7

    # train GAN
    discriminator.train_on_batch(x=true_images,y=np.ones(batch_size))
    discriminator.train_on_batch(x=generated_images,y=np.zeros(batch_size))

    GAN.train_on_batch(noise_gen,disc_out)
    print("\r",i+1," out of ", iters, end="")

# %%
rand = np.random.rand(1000, 100)
# GAN.train_on_batch(rand,[1])
img = generator.predict(np.random.rand(1,100))
img = img.reshape(img_height,img_width,channels)
plt.imshow(img)

plt.imsave("fig.png",img,dpi=300)

discriminator.save("discriminator_model.hdf5")
generator.save("generator_model.hdf5")
