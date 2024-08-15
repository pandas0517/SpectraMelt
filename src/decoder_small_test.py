from decoder import create_decoder
import numpy as np
import tensorflow as tf

tdictionary = np.load("test_sets/dictionary.npy")
ttest_set = np.load("test_sets/test_set.npy")
# tdecoder_test_set = np.load("test_sets/decoder_test_set.npy")
tencoded_test_set = np.load("test_sets/encoded_test_set.npy")
tdic_test_set = np.load("test_sets/dic_test_set.npy")
# decoded_mag_model = tf.keras.models.load_model('models/decoder_mag_model.keras')
# decoded_ang_model = tf.keras.models.load_model('models/decoder_ang_model.keras')
# for idx, test_data in enumerate(ttest_set):


decoder = create_decoder(tdictionary, ttest_set, tdic_test_set, tencoded_test_set)
# decoder = create_decoder(tdictionary, ttest_set, tdecoder_test_set, tencoded_test_set)


# decoded_mag_model.fit(np.abs(tencoded_test_set), np.abs(ttest_set),
#             epochs=20,
#             batch_size=256,
#             shuffle=True,
#             validation_data=(np.abs(tencoded_test_set), np.abs(ttest_set)))
# decoded_ang_model.fit(np.angle(tencoded_test_set), np.angle(ttest_set),
#             epochs=20,
#             batch_size=256,
#             shuffle=True,
#             validation_data=(np.angle(tencoded_test_set), np.angle(ttest_set)))
# decoded_mag_model.save("decoder_mag_model.keras", overwrite=True)
# decoded_ang_model.save("decoder_ang_model.keras", overwrite=True)