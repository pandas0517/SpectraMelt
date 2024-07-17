'''
Created on Jul 11, 2024

@author: pete
'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from itertools import combinations

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    # from tensorflow.keras import layers, losses
    # from tensorflow.keras.datasets import fashion_mnist
    # from tensorflow.keras.models import Model

    # (x_train, _), (x_test, _) = fashion_mnist.load_data()

    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.

    # print (x_train.shape)
    # print (x_test.shape)
    signal_dim = 1000
    num_of_pos_bins = int(signal_dim / 2)
    num_of_sig = 4
    num_of_pos_sig = int(num_of_sig / 2)
    bins = list(range(1, num_of_pos_bins+1))
    comb_pos_sig_size_n = np.array(list(combinations(bins, num_of_pos_sig)))
    comb_neg_sig_size_n = np.copy(-1*comb_pos_sig_size_n)
    num_sig_comb = comb_neg_sig_size_n.size + comb_pos_sig_size_n.size
    sig_array = np.zeros((signal_dim,1,num_sig_comb))
    # comb_neg_sig_size_n = [ x * -1 for x in comb_pos_sig_size_n ]
    comb_pos_sig_size_n.append(comb_neg_sig_size_n)
    pass
    class Autoencoder(Model):
        def __init__(self, latent_dim, shape):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.shape = shape
            self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
            ])
            self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
            layers.Reshape(shape)
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    # shape = x_test.shape[1:]

    latent_dim = 64
    autoencoder = Autoencoder(latent_dim, shape)

    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # autoencoder.fit(x_train, x_train,
    #             epochs=10,
    #             shuffle=True,
    #             validation_data=(x_test, x_test))