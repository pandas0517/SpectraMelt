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
from scipy.fft import fft, ifft
import numpy as np
import random
from itertools import combinations
from signals import multi_tone_sine_wave
import keras
from keras import layers
from signals import generate_LO
from signals import filter_signal
from signals import downsample

def create_test_set(dictionary, t, system_params, wave_params, filter_params, LO_params):
    enc_dim = (dictionary.shape)[0]
    signal_dim = (dictionary.shape)[1]
    wbf_cut_freq = system_params['adc_clock_freq'] * system_params['wbf_cut_mod']
    # num_of_pos_bins = int(signal_dim / 2)
    # num_of_sig = 2
    num_of_pos_sig = 0
    for wave_param in wave_params:
        if wave_param['amp']!= 0:
            num_of_pos_sig += 1
    # num_of_tot_sig = int(num_of_pos_sig * 2)
    pos_bins = list(range(1, wbf_cut_freq))
    # sig_array = np.zeros((signal_dim,1,1))
    test_freq_tot_list = []
    # tot_num_freq_combos = 0
    for total_active_sig in range(1,num_of_pos_sig + 1):
        # test_freq_tot_list.append(list(combinations(pos_bins, total_active_sig)))
        test_freq_tot_list += list(combinations(pos_bins, total_active_sig))
        # tot_num_freq_combos += len(test_freq_tot_list[total_active_sig - 1])
    random.shuffle(test_freq_tot_list)
    tot_num_freq_combos = len(test_freq_tot_list)
    test_sig_set = np.zeros((tot_num_freq_combos,signal_dim), dtype='complex128')

    count = 0
    # for test_freq_list in test_freq_tot_list:
    for test_freq in test_freq_tot_list:
        test_wave_params = []
        for freq in test_freq:
            test_wave_param = {'amp': 0.1,
                              'freq': freq,
                              'phase': 0}
            # test_wave_param['amp'] = 0.1
            # test_wave_param['freq'] = freq
            test_wave_params.append(test_wave_param)
        x, t_test, num_tones = multi_tone_sine_wave(system_params, test_wave_params, filter_params)
        xf = fft(x)
        test_sig_set[count,:] = xf
        test_wave_params.clear()
        count += 1
    encoded_test_set = np.zeros((tot_num_freq_combos,enc_dim),dtype=np.complex128)
    dic_test_set = np.zeros((tot_num_freq_combos,signal_dim),dtype=np.complex128)
    pseudo_inv = np.linalg.pinv(dictionary)
    # test_sig_set = test_sig_set[np.random.permutation(tot_num_freq_combos),:]
    for idx, test_data in enumerate(test_sig_set):
        LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
        y_mixed = np.copy(ifft(test_data)*rising_zero_crossings)
        y_filtered, filt_freq, filt_freq_down = filter_signal(y_mixed, t, filter_params, system_params)
        y_sampled, LO_mod_sampled, t_sampled, tf_sampled, filt_sampled, downsample_train = downsample(y_filtered, LO_mod, t, system_params, rising_zero_crossings, filt_freq)
        encoded_test_set[idx,:] = y_sampled
        dic_test_data = np.dot(pseudo_inv,y_sampled)
        # dic_test_data = np.matmul(dictionary,test_data)
        dic_test_set[idx,:] = dic_test_data
        pass
    return test_sig_set, encoded_test_set, dic_test_set
        # new_freq_list = list(combinations(pos_bins, total_active_sig))
        # test_freq_list = test_freq_list + new_freq_list
        # new_freq_list.clear()
    # x, t, num_tones = multi_tone_sine_wave(system_params, wave_params, filter_params)
    # sig_array = np.zeros((1,signal_dim))
    # for sig in range(1,num_of_pos_sig + 1):
    #     comb_pos_sig= np.array(list(combinations(bins, sig)))
    #     comb_neg_sig = np.copy(-1*comb_pos_sig)
    #     comb_sig = np.stack((comb_neg_sig, comb_pos_sig), axis=2)
    #     comb_sig_shift = np.copy(num_of_pos_bins + comb_sig)
    #     # sig_array_comb_n = np.zeros((signal_dim,1,comb_sig.shape[0]))
    #     sig_array_comb_n = np.zeros((comb_sig.shape[0],signal_dim))
    #     for n, comb in enumerate(comb_sig_shift):
    #         # sig_array_comb_n[comb, 0, n] = 1
    #         # sig_array_comb_n[n, comb] = 1
    #         sig_array_comb_n[n, comb] = num_of_pos_bins
    #     sig_array = np.concatenate((sig_array, sig_array_comb_n), axis=0)
    #     if (sig == 1):
    #         sig_array = np.delete(sig_array,(0), axis=0)
    # pass
    # # sig_array = np.delete(sig_array,0,2)
    # return sig_array

def create_decoder(dictionary, test_set, dic_test_set):
    # This is the size of our encoded representations
    enc_dim = (dictionary.shape)[0]
    dec_dim = (dictionary.shape)[1]
    test_set_size = (test_set.shape)[0]
    train_split = int(0.5 * test_set_size)
    # # encoded_test_set = np.zeros((enc_dim,1,test_set_size),dtype=np.complex128)
    # encoded_test_set = np.zeros((test_set_size,enc_dim),dtype=np.complex128)
    # # test_set = test_set[:,:,np.random.permutation(test_set_size)]
    # test_set = test_set[np.random.permutation(test_set_size),:]
    # for idx, test_data in enumerate(test_set):
    #     # encoded_test_data = np.matmul(dictionary,test_data.T)
    #     LO_mod, rising_zero_crossings, LO, sample_train, sample_train_fast, clock_ticks = generate_LO(t, LO_params, system_params)
    #     encoded_test_data = np.matmul(dictionary,test_data)
    #     # encoded_test_set[:,:,idx] = encoded_test_data
    #     encoded_test_set[idx,:] = encoded_test_data
    #     pass

    # encoded_complex_input_layer = keras.Input(shape=(enc_dim,))
    # encoded_real_input_layer = keras.Input(shape=(enc_dim,))
    # encoded_imag_input_layer = keras.Input(shape=(enc_dim,))
    # decoded_complex = layers.Dense(dec_dim, activation='sigmoid')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='sigmoid')(encoded_imag_input_layer)
    # decoded_real = layers.Dense(dec_dim, activation='relu')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='relu')(encoded_imag_input_layer)
    # decoded_real = layers.Dense(dec_dim, activation='softmax')(encoded_real_input_layer)
    # decoded_imag = layers.Dense(dec_dim, activation='softmax')(encoded_imag_input_layer)
    # decoder_complex = keras.Model(encoded_complex_input_layer, decoded_complex)
    # decoder_real = keras.Model(encoded_real_input_layer, decoded_real)
    # decoder_imag = keras.Model(encoded_imag_input_layer, decoded_imag)
    # decoder_real.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_imag.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_real.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_imag.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_complex.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_real.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder_imag.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder.fit(np.real(encoded_test_set[:,:,:train_split]), np.real(test_set[:,:,:train_split]),
    # decoder_complex.fit(encoded_test_set[:train_split,:], test_set[:train_split,:],
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(encoded_test_set[train_split:,:], test_set[train_split:,:]))
    # decoder_real.fit(np.real(encoded_test_set[:train_split,:]), np.real(test_set[:train_split,:]),
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(np.real(encoded_test_set[train_split:,:]), np.real(test_set[train_split:,:])))
    # decoder_imag.fit(np.imag(encoded_test_set[:train_split,:]), np.imag(test_set[:train_split,:]),
    #             epochs=50,
    #             batch_size=256,
    #             shuffle=True,
    #             # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
    #             validation_data=(np.imag(encoded_test_set[train_split:,:]), np.imag(test_set[train_split:,:])))
    # decoder_complex.save("decoder_complex.keras", overwrite=True)
    # decoder_real.save("decoder_real.keras", overwrite=True)
    # decoder_imag.save("decoder_imag.keras", overwrite=True)

    encoded_mag_input_layer = keras.Input(shape=(enc_dim,))
    encoded_ang_input_layer = keras.Input(shape=(dec_dim,))
    # decoded_mag = layers.Dense(dec_dim, activation='sigmoid')(encoded_real_input_layer)
    # decoded_ang = layers.Dense(dec_dim, activation='sigmoid')(encoded_imag_input_layer)
    # decoded_mag = layers.Dense(dec_dim, activation='relu')(encoded_mag_input_layer)
    # decoded_ang = layers.Dense(dec_dim, activation='relu')(encoded_ang_input_layer)
    decoded_mag_model = keras.Sequential()
    # decoded_mag_model.add(keras.Input(shape=(enc_dim,)))
    decoded_mag_model.add(keras.Input(shape=(dec_dim,)))
    # decoded_mag_model.add(layers.Dense(int(dec_dim/5), name="layer1", kernel_initializer=keras.initializers.Zeros()))
    decoded_mag_model.add(layers.Dense(int(dec_dim), name="layer1"))
    # decoded_mag_model.add(layers.Dense(int(dec_dim), name="layer2"))
    # decoded_mag_model.add(layers.Dense(int(dec_dim), name="layer5", kernel_initializer=keras.initializers.Zeros()))
    decoded_mag_model.add(layers.Dense(dec_dim, name="decoded_mag_model_out"))
    decoded_mag_model.summary()
    # decoded_mag = layers.Dense(dec_dim, activation='softmax')(encoded_mag_input_layer)
    decoded_ang = layers.Dense(dec_dim)(encoded_ang_input_layer)
    # decoder_mag = keras.Model(encoded_mag_input_layer, decoded_mag)
    # decoder_mag = keras.Model(inputs=decoded_mag_model.inputs, outputs=decoded_mag_model.outputs)
    decoder_ang = keras.Model(encoded_ang_input_layer, decoded_ang)
    # decoder_real.compile(optimizer='adam', loss='binary_crossentropy')
    # decoder_imag.compile(optimizer='adam', loss='binary_crossentropy')
    opt = keras.optimizers.Adam(learning_rate=0.01)
    decoded_mag_model.compile(optimizer=opt, loss='mean_squared_error')
    decoder_ang.compile(optimizer='adam', loss='mean_squared_error')
    # decoder_mag.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder_ang.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    # decoder.fit(np.real(encoded_test_set[:,:,:train_split]), np.real(test_set[:,:,:train_split]),
    # decoded_mag_model.fit(np.abs(encoded_test_set[:train_split,:]), np.abs(test_set[:train_split,:]),
    decoded_mag_model.fit(np.abs(dic_test_set), np.abs(test_set),
                epochs=20,
                batch_size=256,
                shuffle=True,
                # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
                # validation_data=(np.abs(encoded_test_set[train_split:,:]), np.abs(test_set[train_split:,:])))
                validation_data=(np.abs(dic_test_set), np.abs(test_set)))
    decoder_ang.fit(np.angle(dic_test_set[:train_split,:]), np.angle(test_set[:train_split,:]),
                epochs=20,
                batch_size=256,
                shuffle=True,
                # validation_data=(np.real(encoded_test_set[:,:,train_split:]), np.real(test_set[:,:,train_split:])))
                validation_data=(np.angle(dic_test_set[train_split:,:]), np.angle(test_set[train_split:,:])))
    decoded_mag_model.save("decoder_mag_model.keras", overwrite=True)
    # decoder_mag.save("decoder_mag.keras", overwrite=True)
    decoder_ang.save("decoder_ang.keras", overwrite=True)
    pass