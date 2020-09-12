from PIL import Image
from os import listdir
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
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
#
# i = 0
# j = 0
# # Image.open('Cat/cats/'+listdir('./Cat/cats')[0])
#
# for filename in listdir('./Cat/cats'):
#     i += 1
#     if filename.endswith('.jpg'):
#         try:
#             # open the image file
#             image = tf.io.read_file('Cat/cats/'+filename)
#             # verify that it is, in fact an image
#             image = tf.image.decode_image(image, channels=3)
#         except:
#             j += 1
#             # os.remove('Cat/cats/'+filename)
# print(j)

# %%
iters = int(sys.argv[1])
# iters = 2
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
data_dir = pathlib.Path('Cat')
images = list(data_dir.glob('*.jpe'))

batch_size = 32
img_height = 64
img_width = 64
channels = 3
buffer_size = 100

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Cat', validation_split=0.2, subset="training", color_mode='rgb', seed=123, image_size=(img_height, img_width), label_mode=None, batch_size=1, shuffle=True)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Cat', validation_split=0.2, subset="validation", color_mode='rgb', seed=123, image_size=(img_height, img_width),  label_mode=None, batch_size=1, shuffle=True)

train_ds = train_ds.unbatch().batch(
    32, drop_remainder=True)

for element in train_ds.take(2).as_numpy_iterator():
    image = element[0]

plt.imshow(image.reshape(img_height, img_width, 3)
           / max(np.concatenate(np.concatenate(image))), 'Greys')

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
optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
# discriminator.summary()

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
# generator.summary()

# %%
discriminator.trainable = False
z = keras.Input(shape=(100,))
gen = generator(z)
dis = discriminator(gen)
GAN = keras.Model(z, dis)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer,
            metrics=['accuracy'])
# GAN.summary()

# %%
bl = 1
err = [[], []]
for i in range(iters):
    # iterator = iter(train_ds)
    print("\r", i+1, " out of ", iters, end="")
    print("\n")
    for true_images in train_ds:
        bl += 1
        # set batch size
        batch_size = len(true_images)

        # generate images
        noise_gen = np.random.rand(batch_size, 100)
        generated_images = generator.predict(noise_gen)

        # get discriminator prediction
        disc_out = np.random.rand(batch_size, 1)*0.5+0.7

        # train GAN
        discriminator.train_on_batch(x=true_images, y=np.ones(batch_size))
        d_loss = discriminator.train_on_batch(
            x=generated_images, y=np.zeros(batch_size))

        g_loss = GAN.train_on_batch(noise_gen, disc_out)
        err[0].append(g_loss[0])
        err[1].append(d_loss[0])
        print("\r", 'batch: ', bl, " g_loss: ",
              g_loss, ', d_loss: ', d_loss, end="")

plt.plot(np.array(err[0]), label='generator loss')
plt.plot(np.array(err[1]), label='discriminator loss')
plt.legend()
plt.savefig('err.png')

# %%
rand = np.random.rand(1000, 100)
# GAN.train_on_batch(rand,[1])
img = generator.predict(np.random.rand(1, 100))
img = img.reshape(img_height, img_width, channels)
plt.imshow(img)

plt.imsave("fig.png", img, dpi=300)

discriminator.save("discriminator_model.hdf5")
generator.save("generator_model.hdf5")
