import keras
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras import layers
from keras import losses
import tensorflow as tf
from numpy import sin, cos
import numpy as np
from utility import load_settings, replace_extension
from scipy.fftpack import fft
import os

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(losses.mean_squared_error(y_true, y_pred))

def reset_tensforflow_session():
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()

def model_prediction(init_guess, file_path, mode, aux_file_path=None):
    coef_real = 0
    coef_imag = 0
    mlp_model = tf.keras.models.load_model(file_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    use_aux_model = False
    if aux_file_path is not None:
        if os.path.isfile(aux_file_path):
            use_aux_model = True
            mlp_model_aux = tf.keras.models.load_model(aux_file_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    if ( mode == 'complex' ):
        x_pre_real = np.real(init_guess)
        x_pre_imag = np.imag(init_guess)
        x_pre_flattened = np.concatenate((x_pre_real, x_pre_imag))
        coef_predict = mlp_model.predict(x_pre_flattened.reshape(1, x_pre_flattened.shape[0]))
        coef_flattened = coef_predict.reshape(-1)
        coef_split = np.split(coef_flattened, 2)
        coef_real = coef_split[0]
        coef_imag = coef_split[1]
    else:
        if ( mode == 'real_imag' ):
            x_pre_real = np.real(init_guess)
            x_pre_imag = np.imag(init_guess)
            x_pre_real_reshape = x_pre_real.reshape((1,x_pre_real.shape[0]))
            x_pre_imag_reshape = x_pre_imag.reshape((1,x_pre_imag.shape[0]))
            coef_predict_real = mlp_model.predict(x_pre_real_reshape)
            coef_real = coef_predict_real.reshape(-1)
            if use_aux_model:
                coef_predict_imag = mlp_model_aux.predict(x_pre_imag_reshape)
                coef_imag = coef_predict_imag.reshape(-1)
    
        elif ( mode == 'mag_ang' ):
            x_pre_mag = np.abs(init_guess)
            x_pre_ang = np.angle(init_guess)
            x_pre_mag_reshape = x_pre_mag.reshape((1,x_pre_mag.shape[0]))
            x_pre_ang_reshape = x_pre_ang.reshape((1,x_pre_ang.shape[0]))
            coef_predict_mag = mlp_model.predict(x_pre_mag_reshape)
            coef_predict_ang = 0
            if use_aux_model:
                coef_predict_ang = mlp_model_aux.predict(x_pre_ang_reshape)
                
            coef_real = (coef_predict_mag*cos(coef_predict_ang)).reshape(-1)
            coef_imag = (coef_predict_mag*sin(coef_predict_ang)).reshape(-1)
        elif ( mode == 'active_zones' ):
            x_pre_mag = np.abs(init_guess)
            x_pre_ang = np.angle(init_guess)
            x_pre_mag_reshape = x_pre_mag.reshape((1,x_pre_mag.shape[0]))
            x_pre_ang_reshape = x_pre_ang.reshape((1,x_pre_ang.shape[0]))
            coef_predict_mag = mlp_model.predict(x_pre_mag_reshape)
            coef_predict_ang = 0
            coef_real = (coef_predict_mag*cos(coef_predict_ang)).reshape(-1)
            coef_imag = (coef_predict_mag*sin(coef_predict_ang)).reshape(-1)
    return coef_real, coef_imag

def set_training_params(training_params=None, training_conf=None):
    if training_conf is None:
        if training_params is None:
            training_params = {
                "processing_system": "system1",
                "modes": [
                    "mag"
                ],
                "total_num_sigs": 40000,
                "train_test_split_percentage": 0.7,
                "loss_type": "root_mean_squared_error",
                "learning_rate": 0.00001,
                "num_epochs": 200,
                "batch_sz": 128,
                "pre_multiply": 0.01,
                "save_fft_file": True,
                "save_active_zones_file": True,
                "save_premultiply": True,
                "use_fft": False,
                "active_zones_min_mag": 500,
                "use_active_zones": True,
                "pre_omp": False,
                "early_stopping": {
                    "monitor": "val_loss",
                    "min_delta": 0.1,
                    "patience": 4,
                    "verbose": 1,
                    "start_from_epoch": 5,
                    "restore_best_weights" :True
                }
            }
    else:
        training_params = load_settings(training_conf)
    return training_params

def set_test_train(train_size,
                   output_sig_set,
                   use_premultiply,
                   NYFR_test_harness,
                   dictionary_file_path,
                   training_params,
                   system_params,
                   mode,
                   premultiply_file_path):
    dictionary = np.load(dictionary_file_path)
    premultiply_sig_set = []
    premultiply_sig_set_train = []
    premultiply_sig_set_test = []
    for i, output_signal in enumerate(output_sig_set):
        if ( use_premultiply ):
            premultiply_signal = np.copy(output_signal)
        else:
            if ( training_params['pre_omp'] ):
                original_recovery = system_params['recovery']
                system_params['recovery'] = 'c_omp'
                NYFR_test_harness.set_system_params(system_params=system_params)
                premultiply_signal = NYFR_test_harness.recover_signal(training_params['pre_multiply']*dictionary, output_signal)
                system_params['recovery'] = original_recovery
                NYFR_test_harness.set_system_params(system_params=system_params)
            else:
                pseudo = np.linalg.pinv( training_params['pre_multiply'] * dictionary)
                premultiply_signal = np.dot(pseudo,output_signal)

            premultiply_sig_set.append(premultiply_signal)

        if ( mode == 'real'):
            premultiply_sig = np.copy(premultiply_signal.real)
        elif ( mode == 'imag'):
            premultiply_sig = np.copy(premultiply_signal.imag)
        elif ( mode == 'mag'):
            premultiply_sig = np.abs(premultiply_signal)
        elif ( mode == 'ang'):
            premultiply_sig = np.angle(premultiply_signal)
        elif ( mode == 'complex'):
            premultiply_sig = np.concatenate((premultiply_signal.real, premultiply_signal.imag))    

        if ( i < train_size ):
                premultiply_sig_set_train.append(premultiply_sig)
        else:
                premultiply_sig_set_test.append(premultiply_sig)

    if ( not use_premultiply and training_params['save_premultiply'] ):
        premultiply_sig_set_array = np.array(premultiply_sig_set)
        np.save(premultiply_file_path, premultiply_sig_set_array)

    return np.array(premultiply_sig_set_train), np.array(premultiply_sig_set_test)

def create_model(mlp_model_file_path,
                 output_file_path,
                 recovery_log_file_path,
                 model_log_file_path,
                 model_input_size,
                 training_params,
                 premultiply_sig_set_train,
                 premultiply_sig_set_test,
                 output_sig_set_train,
                 output_sig_set_test):

    if ( os.path.isfile( mlp_model_file_path )):
        mlp_model = tf.keras.models.load_model(mlp_model_file_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    else:
        mlp_model = keras.Sequential()
        mlp_model.add(keras.Input(shape=(premultiply_sig_set_train.shape[1],)))
        mlp_model.add(layers.Dense(premultiply_sig_set_train.shape[1], name="mlp_model_layer_1"))
        mlp_model.add(layers.Dense(premultiply_sig_set_train.shape[1], name="mlp_model_out"))
        
        # mlp_model.add(layers.Reshape((NYFR_test_harness.get_Zones(), NYFR_test_harness.get_K_band()), input_shape=(model_input_size,)))
        # mlp_model.add(layers.Conv1D(filters=NYFR_test_harness.get_Zones(),
        #                             kernel_size=NYFR_test_harness.get_K_band(),
        #                             padding='same',
        #                             input_shape=(NYFR_test_harness.get_Zones(),NYFR_test_harness.get_K_band()),
        #                             # activity_regularizer=regularizers.l1(0.001),
        #                             name="mlp_model_layer_1"))
        # mlp_model.add(layers.Flatten())
        # mlp_model.add(layers.Dense(model_input_size, activity_regularizer=regularizers.l1(0.01), name="mlp_model_layer_2"))
        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_3"))
        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_4"))
        # mlp_model.add(layers.Dense(NYFR_test_harness.get_Zones(),
        #                             activation='softmax',
        #                         #    activity_regularizer=regularizers.l2(0.001),
        #                             name="mlp_model_out"))
        # mlp_model.add(layers.Activation('relu'))
        
        mlp_opt = keras.optimizers.Adam(learning_rate=training_params['learning_rate'])
        if training_params['loss_type'] == "root_mean_squared_error":
            mlp_model.compile(optimizer=mlp_opt, loss=root_mean_squared_error)
        else:
            mlp_model.compile(optimizer=mlp_opt, loss=training_params['loss_type'])

    early_stopping = EarlyStopping(monitor=training_params['early_stopping']['monitor'],
                                    min_delta=training_params['early_stopping']['min_delta'],
                                    patience=training_params['early_stopping']['patience'],
                                    verbose=training_params['early_stopping']['verbose'],
                                    start_from_epoch=training_params['early_stopping']['start_from_epoch'],
                                    restore_best_weights=training_params['early_stopping']['restore_best_weights'])
    mlp_model.fit(premultiply_sig_set_train, output_sig_set_train,
                    epochs=training_params['num_epochs'],
                    batch_size=training_params['batch_sz'],
                    shuffle=True,
                    validation_data=(premultiply_sig_set_test, output_sig_set_test),
                    callbacks=[early_stopping])
    mlp_model.save(mlp_model_file_path, overwrite=True)

    with open(model_log_file_path, "a") as model_log:
        model_log.write(output_file_path + "\n")
    with open(recovery_log_file_path, "a") as recovery_log:
        recovery_log.write(output_file_path + "\n")
    reset_tensforflow_session()

def set_recovery_mode(training_params, mode):
    recovery_mode = None
    if training_params['use_active_zones']:
        recovery_mode = "active_zones"
    elif training_params['use_fft']:
        if (mode == 'real'):
            recovery_mode = "real_imag"
        elif (mode == 'imag'):
            recovery_mode = "real_imag"
        elif (mode == 'mag'):
            recovery_mode = "mag_ang"
        elif (mode == 'ang'):
            recovery_mode = "mag_ang"
        elif (mode == 'complex'):
            recovery_mode = "complex"
    return recovery_mode

def create_model_outputs(input_file_path, fft_file_path, active_zones_file_path, zones, training_params, mode, wb_nyquist_rate):
    fft_sig_set = None
    if training_params['use_fft']:
        if ( os.path.isfile(fft_file_path) ):
            fft_sig_set = np.load(fft_file_path)
            if fft_sig_set.shape[0] != training_params['total_num_sigs']:
                fft_sig_set = None

    active_zones_sig_set = None
    if training_params['use_active_zones']:
        if ( os.path.isfile(active_zones_file_path) ):
            active_zones_sig_set = np.load(active_zones_file_path)
            if active_zones_sig_set.shape[0] != training_params['total_num_sigs']:
                active_zones_sig_set = None
    
    output_sig_set = None
    if ( fft_sig_set is None and active_zones_sig_set is None ):
        input_sig_set_total = np.load(input_file_path)
        num_input_sigs_total = input_sig_set_total.shape[0]
        input_sigs_not_used = num_input_sigs_total - training_params['total_num_sigs']
        _, input_sig_set = np.vsplit(input_sig_set_total, [input_sigs_not_used])
        active_zones = np.zeros(zones, dtype="float64")
        fft_sig_list = []
        active_zones_sig_list = []

        for i, input_sig in enumerate(input_sig_set):
            input_sig_fft = fft(input_sig)/(2*wb_nyquist_rate)
            fft_sig_list.append(input_sig_fft)
            input_zones = np.array_split(np.abs(input_sig_fft), zones)
            # non_zero_in_zones = [np.any(zone != 0) for zone in input_zones]
            for i, zone in enumerate(input_zones):
                if np.any( zone > training_params['active_zones_min_mag'] ):
                    active_zones[i] = 1
            active_zones_sig_list.append(np.copy(active_zones))
            active_zones.fill(0)

        fft_sig_set = np.array(fft_sig_list)
        if ( training_params['save_fft_file'] ):
            if ( not os.path.exists(fft_file_path) ):
                np.save(fft_file_path, fft_sig_set)
        
        active_zones_sig_set = np.array(active_zones_sig_list)
        if ( training_params['save_active_zones_file'] ):
            if ( not os.path.exists(active_zones_file_path) ):
                np.save(active_zones_file_path, active_zones_sig_set)

    if training_params['use_active_zones']:
        output_sig_set = active_zones_sig_set
    elif training_params['use_fft']:
        output_sig_list = []
        for i, fft_sig in enumerate(fft_sig_set):
            if (mode == 'real'):
                output_sig_list.append(fft_sig.real)
            elif (mode == 'imag'):
                output_sig_list.append(fft_sig.real)
            elif (mode == 'mag'):
                output_sig_list.append(np.abs(fft_sig))
            elif (mode == 'ang'):
                output_sig_list.append(np.angle(fft_sig))
            elif (mode == 'complex'):
                output_sig_list.append(np.concatenate((fft_sig.real, fft_sig.imag)))
        output_sig_set = np.array(output_sig_list)

    model_input_size = output_sig_set.shape[1]
    num_input_sigs = output_sig_set.shape[0]
    return output_sig_set, model_input_size, num_input_sigs

def create_mlp1_models(NYFR_test_harness, training_params=None, training_conf=None):
    training_params = set_training_params(training_params=training_params, training_conf=training_conf)
    if training_params['use_fft'] and training_params['use_active_zones']:
        training_params['use_fft'] = False
    training_params["pre_multiply"] = 4 / NYFR_test_harness.get_adc_clock_freq()
    directories = NYFR_test_harness.get_directories()
    if directories is None:
        print("NYFR Test Harness must be initialized")
    files = NYFR_test_harness.get_filenames()
    recovery_params = NYFR_test_harness.get_recovery_params()
    dictionary_params = NYFR_test_harness.get_dictionary_params()
    system_params = NYFR_test_harness.get_system_params()
    input_set_params = NYFR_test_harness.get_input_set_params()
    for mode in training_params['modes']:
        for noise_level, _ in input_set_params["noise_levels"]:
            for phase_shift, _ in input_set_params["phase_shifts"]:
                for input_tones, _ in input_set_params["input_tones"]:
                    fft_file_path = os.path.join(directories['fft'],
                                                 noise_level,
                                                 phase_shift,
                                                 files["input_tones"][input_tones]["sigs"])
                    input_file_path = os.path.join(directories["input"],
                                                   noise_level,
                                                   phase_shift,
                                                   files["input_tones"][input_tones]["sigs"])
                    active_zones_file_path = os.path.join(directories['active_zones'],
                                                 noise_level,
                                                 phase_shift,
                                                 files["input_tones"][input_tones]["sigs"])
                    model_log_file_path = os.path.join(directories['mlp_models'][dictionary_params['version']][mode],
                                                       files['mlp_models']['log'][training_params['processing_system']])

                    output_sig_set, model_input_size, num_input_sigs = create_model_outputs(input_file_path,
                                                                                            fft_file_path,
                                                                                            active_zones_file_path,
                                                                                            NYFR_test_harness.get_Zones(),
                                                                                            training_params,
                                                                                            mode,
                                                                                            NYFR_test_harness.get_wb_nyquist_rate())
                    recovery_mode = set_recovery_mode(training_params, mode)

                    train_size = int(num_input_sigs * training_params['train_test_split_percentage'])
                    # test_size = num_input_sigs - train_size

                    output_sig_set_train, output_sig_set_test = np.vsplit(output_sig_set, [train_size])
                    output_sig_set = None


                    if recovery_mode == "active_zones" or recovery_mode == "complex":
                        recovery_log_file_path = os.path.join(directories['recovery'][dictionary_params['version']][recovery_params['type']],
                                                              recovery_mode,
                                                              files['recovery'][recovery_mode][training_params['processing_system']])
                    else:
                        recovery_log_file_path = os.path.join(directories['recovery'][dictionary_params['version']][recovery_params['type']],
                                                              recovery_mode,
                                                              files['recovery'][recovery_mode][mode][training_params['processing_system']])
                    for f_mod, _ in input_set_params["f_mods"]:
                        for f_delta, _ in input_set_params["f_deltas"]:
                            dictionary_file_path = os.path.join(directories['dictionary'][dictionary_params['version']],
                                                                f_mod,
                                                                f_delta,
                                                                files['dictionary']['name'])
                            output_file_path = os.path.join(directories['output'],
                                                            noise_level,
                                                            phase_shift,
                                                            f_mod,
                                                            f_delta,
                                                            files["input_tones"][input_tones]["sigs"])
                            premultiply_file_path = os.path.join(directories['premultiply'][dictionary_params['version']],
                                                                 noise_level,
                                                                 phase_shift,
                                                                 f_mod,
                                                                 f_delta,
                                                                 files["input_tones"][input_tones]["sigs"])
                            if ( input_set_params["use_per_signal_model"] ):
                                mlp_model_file_path = os.path.join(directories['mlp_models'][dictionary_params['version']][mode],
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                files["input_tones"][input_tones]["sigs"])
                                mlp_model_file_path = replace_extension(mlp_model_file_path, "keras")
                            else:
                                mlp_model_file_path = os.path.join(directories['mlp_models'][dictionary_params['version']][mode],
                                                                noise_level,
                                                                phase_shift,
                                                                f_mod,
                                                                f_delta,
                                                                files['mlp_models']['name'])
                                
                            found_string_in_file = False
                            with open(model_log_file_path, "r") as model_log:
                                for line in model_log:
                                    if output_file_path in line:
                                        found_string_in_file = True
                                        break

                            if not found_string_in_file:
                                use_premultiply = False
                                if ( os.path.isfile(premultiply_file_path) ):
                                    output_sig_set = np.load(premultiply_file_path)
                                    if output_sig_set.shape[0] == training_params['total_num_sigs']:
                                        use_premultiply = True

                                if not use_premultiply:
                                    
                                    output_sig_set_total = np.load(output_file_path)
                                    num_input_sigs_total = output_sig_set_total.shape[0]
                                    output_sigs_not_used = num_input_sigs_total - training_params['total_num_sigs']
                                    _, output_sig_set = np.vsplit(output_sig_set_total, [output_sigs_not_used])
                                    del output_sig_set_total

                                premultiply_sig_set_train, premultiply_sig_set_test = set_test_train(train_size,
                                                                                                    output_sig_set,
                                                                                                    use_premultiply,
                                                                                                    NYFR_test_harness,
                                                                                                    dictionary_file_path,
                                                                                                    training_params,
                                                                                                    system_params,
                                                                                                    mode,
                                                                                                    premultiply_file_path)
                                
                                create_model(mlp_model_file_path,
                                            output_file_path,
                                            recovery_log_file_path,
                                            model_log_file_path,
                                            model_input_size,
                                            training_params,
                                            premultiply_sig_set_train,
                                            premultiply_sig_set_test,
                                            output_sig_set_train,
                                            output_sig_set_test)