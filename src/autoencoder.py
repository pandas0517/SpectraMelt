'''
Created on Jul 11, 2024

@author: pete
'''
# class MyClass(object):
#     '''
#     classdocs
#     '''
#
#
#     def __init__(self, params):
#         '''
#         Constructor
#         '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
# import tensorflow as tf
from itertools import combinations
import keras
from keras import layers
from keras.datasets import mnist

def create_test_set(dictionary, system_params, wave_params):
    # signal_dim = 1000
    signal_dim = (dictionary.shape)[1]
    wb_cut_freq = system_params['adc_clock_freq'] * system_params['wb_filt_cut_freq']
    num_of_pos_bins = int(signal_dim / 2)
    # num_of_sig = 2
    num_of_pos_sig = 0
    for wave_param in wave_params:
        if wave_param['amp']!= 0:
            num_of_pos_sig += 1
    # num_of_tot_sig = int(num_of_pos_sig * 2)
    bins = list(range(0, wb_cut_freq))
    sig_array = np.zeros((signal_dim,1,1))
    for sig in range(1,num_of_pos_sig + 1):
        comb_pos_sig= np.array(list(combinations(bins, sig)))
        comb_neg_sig = np.copy(-1*comb_pos_sig)
        comb_sig = np.stack((comb_neg_sig, comb_pos_sig), axis=2)
        comb_sig_shift = np.copy(num_of_pos_bins + comb_sig)
        sig_array_comb_n = np.zeros((signal_dim,1,comb_sig.shape[0]))
        for n, comb in enumerate(comb_sig_shift):
            sig_array_comb_n[comb, 0, n] = 1
        sig_array = np.concatenate((sig_array, sig_array_comb_n), axis=2)
        if (sig == 1):
            sig_array = np.delete(sig_array,0,2)

    # sig_array = np.delete(sig_array,0,2)
    return sig_array

def create_autoencoder(dictionary, test_set):
    # This is the size of our encoded representations
    enc_dim = (dictionary.shape)[0]
    dec_dim = (dictionary.shape)[1]
    test_set_size = (test_set.shape)[2]
    train_test_split = int(0.3 * test_set_size)
    encoded_test_set = np.zeros((enc_dim,1,test_set_size))
    test_set = test_set[:,:,np.random.permutation(test_set_size)]
    for idx, test_data in enumerate(test_set.T):
        encoded_test_data = np.matmul(dictionary,test_data.T)
        encoded_test_set[:,:,idx] = encoded_test_data
        pass
    train_set = 0
    encoded_train_set = 0
    encoded_input_layer = keras.Input(shape=(enc_dim,))
    decoded = layers.Dense(dec_dim, activation='sigmoid')(encoded_input_layer)
    decoder = keras.Model(encoded_input_layer, decoded)
    decoder.compile(optimizer='adam', loss='binary_crossentropy')
    decoder.fit(encoded_train_set, train_set,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(encoded_test_set, test_set))     
    pass
    # encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    # # This is our input image
    # input_img = keras.Input(shape=(784,))
    # # "encoded" is the encoded representation of the input
    # encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(784, activation='sigmoid')(encoded)

    # # This model maps an input to its reconstruction
    # autoencoder = keras.Model(input_img, decoded)    
    # # This model maps an input to its encoded representation
    # encoder = keras.Model(input_img, encoded)
    # # This is our encoded (32-dimensional) input
    # encoded_input = keras.Input(shape=(encoding_dim,))
    # # Retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # Create the decoder model
    # decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # (x_train, _), (x_test, _) = mnist.load_data()
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # print(x_train.shape)
    # print(x_test.shape)    
    # autoencoder.fit(x_train, x_train,
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             validation_data=(x_test, x_test))   
    # # Encode and decode some digits
    # # Note that we take them from the *test* set
    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)

    # Use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Model

# (x_train, _), (x_test, _) = fashion_mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# print (x_train.shape)
# print (x_test.shape)
# class Autoencoder(Model):
#     def __init__(self, latent_dim, shape):
#         super(Autoencoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.shape = shape
#         self.encoder = tf.keras.Sequential([
#         layers.Flatten(),
#         layers.Dense(latent_dim, activation='relu'),
#         ])
#         self.decoder = tf.keras.Sequential([
#         layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
#         layers.Reshape(shape)
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# shape = x_test.shape[1:]

# latent_dim = 64
# autoencoder = Autoencoder(latent_dim, shape)

# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
# autoencoder.fit(x_train, x_train,
#             epochs=10,
#             shuffle=True,
#             validation_data=(x_test, x_test))
# encoded_imgs = autoencoder.encoder(x_test).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i])
#     plt.title("original")
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i])
#     plt.title("reconstructed")
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()