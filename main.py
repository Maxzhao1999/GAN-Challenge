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
# from pstats import SortKey
import timeit
import re
import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
path = '/Users/Max/OneDrive - Imperial College London/4th yr project/GAN-Challenge/scenes'
iters = int(sys.argv[1])

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
dropout = 0.4

discriminator_input = keras.Input(shape=(28, 28, 1))
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
dim = 7
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
x = layers.Conv2DTranspose(1, 5, padding='same')(x)
generator_output = layers.Activation('sigmoid')(x)
generator = keras.Model(generator_input, generator_output, name='generator')
generator.summary()

# %%
# tf.keras.utils.plot_model(generator)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train=x_train.reshape(x_train.shape[0],28,28,1)/255
x_test=x_test.reshape(x_test.shape[0],28,28,1)/255
y_test=np.ones(x_test.shape[0])
y_train = np.ones(x_train.shape[0])
x_fake = np.random.rand(*x_train.shape)
y = np.append(np.ones(x_train.shape[0]),np.zeros(x_fake.shape[0]))
x = np.append(x_train,x_fake,axis=0)

discriminator.fit(x, y, epochs=1, batch_size=32)

# %%
discriminator.trainable = False
z = keras.Input(shape=(100,))
gen = generator(z)
dis = discriminator(gen)
GAN = keras.Model(z,dis)
GAN.compile(loss='binary_crossentropy', optimizer=optimizer,
metrics=['accuracy'])
GAN.summary()
np.zeros((3,3))

# %%
def train_on_n (batch_size=32):
    # generate images from noise
    noise_gen = np.random.rand(batch_size,100)
    generated_images = generator.predict(noise_gen)

    # load real images (for later use)

    # get discriminator prediction
    disc_out = np.ones([batch_size,1])+(np.random.rand(batch_size,1)-0.5)*2

    # train discriminator (for later use)
    # discriminator.train_on_batch(generated_images,np.zeros(generated_images.shape[0]))
    # train GAN
    GAN.train_on_batch(noise_gen,disc_out)

for i in range(iters):
    print("\r",i+1," out of ", iter, end="")
    train_on_n()
    # discriminator.predict
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
rand = np.random.rand(1,100)
# GAN.train_on_batch(rand,[1])
img = generator.predict(np.random.rand(1,100))
img = img.reshape(28,28)
plt.imshow(img)

plt.imsave("fig.png",img,dpi=300)
discriminator.predict(img.reshape(1,28,28,1))
