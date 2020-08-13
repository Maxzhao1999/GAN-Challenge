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
import re
import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
discriminator = tf.keras.models.load_model('discriminator_model.hdf5')
generator = tf.keras.models.load_model('generator_model.hdf5',compile=True)
# %%
rand = np.random.rand(1000,100)
# GAN.train_on_batch(rand,[1])
img = generator.predict(np.random.rand(5,100))
img = img.reshape(5,28,28)
for i in range(len(img)):
    plt.imshow(img[i],cmap='Greys')
    plt.show()

# plt.imsave("fig.png",img,dpi=300)

# eval = generator.predict(rand)
# eval = eval.reshape(1000,28,28,1)
#
# print("discriminator mean: ", np.mean(discriminator.predict(eval)))
