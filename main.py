import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#%%
path='/Users/Max/OneDrive - Imperial College London/4th yr project/GAN-Challenge/scenes'
dataset=keras.preprocessing.image_dataset_from_directory(path,batch_size=64,image_size=(200,200))
#%%
dropout=0.4
discriminator_input = keras.Input(shape=(32,32,3))
discriminator_input.shape
discriminator_input.dtype
x = layers.Conv2D(16,3,strides=2,activation='relu',padding='same')(discriminator_input)
x = layers.Dropout(dropout)(x)
x = layers.Conv2D(32,3,strides=2,activation='relu',padding='same')(x)
x = layers.Dropout(dropout)(x)
x = layers.Conv2D(32,3,strides=2,activation='relu',padding='same')(x)
x = layers.Dropout(dropout)(x)
x = layers.Flatten()(x)
x = layers.Dense(1)(x)
discriminator_output = layers.Activation('sigmoid')(x)
discriminator=keras.Model(discriminator_input,discriminator_output,name='discriminator')
discriminator.summary()
#%%
