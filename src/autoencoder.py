'''
Created on Jul 11, 2024

@author: pete
'''
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # import tensorflow as tf
    from itertools import combinations

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
    signal_dim = 1000
    num_of_pos_bins = int(signal_dim / 2)
    num_of_sig = 4
    num_of_pos_sig = int(num_of_sig / 2)
    bins = list(range(0, num_of_pos_bins))
    sig_array = np.zeros((signal_dim,1,1))
    for sig in range(1,num_of_pos_sig + 1):
        comb_pos_sig= np.array(list(combinations(bins, sig)))
        comb_neg_sig = np.copy(-1*comb_pos_sig)
        comb_sig = np.stack((comb_neg_sig, comb_pos_sig), axis=2)
        comb_sig_shift = np.copy(num_of_pos_bins + comb_sig)
        sig_array_comb_n = np.zeros((signal_dim,1,comb_sig.shape[0]))
        for n, comb in enumerate(comb_sig_shift):
            sig_array_comb_n[comb, 0, n] = 1
            # print(sig_array_comb_n[:,0,n])
        sig_array = np.concatenate((sig_array, sig_array_comb_n), axis=2)
        pass
    sig_array = np.delete(sig_array,0,2)
    # comb_neg_sig_size_n = [ x * -1 for x in comb_pos_sig_size_n ]
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