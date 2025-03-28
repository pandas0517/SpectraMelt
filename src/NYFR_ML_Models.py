import keras
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras import layers
from keras import losses
import tensorflow as tf
from numpy import sin, cos
import numpy as np
from utility import load_settings, get_all_sub_dirs
from utility import get_all_file_paths, get_file_sub_dirs
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
        if aux_file_path is not None:
            mlp_model_aux = tf.keras.models.load_model(aux_file_path)
        pass
        if ( mode == 'real_imag' ):
            x_pre_real = np.real(init_guess)
            x_pre_imag = np.imag(init_guess)
            x_pre_real_reshape = x_pre_real.reshape((1,x_pre_real.shape[0]))
            x_pre_imag_reshape = x_pre_imag.reshape((1,x_pre_imag.shape[0]))
            coef_predict_real = mlp_model.predict(x_pre_real_reshape)
            coef_real = coef_predict_real.reshape(-1)
            if aux_file_path is not None:
                coef_predict_imag = mlp_model_aux.predict(x_pre_imag_reshape)
                coef_imag = coef_predict_imag.reshape(-1)
    
        elif ( mode == 'mag_ang' ):
            x_pre_mag = np.abs(init_guess)
            x_pre_ang = np.angle(init_guess)
            x_pre_mag_reshape = x_pre_mag.reshape((1,x_pre_mag.shape[0]))
            x_pre_ang_reshape = x_pre_ang.reshape((1,x_pre_ang.shape[0]))
            coef_predict_mag = mlp_model.predict(x_pre_mag_reshape)
            coef_predict_ang = 0
            if aux_file_path is not None:
                coef_predict_ang = mlp_model_aux.predict(x_pre_ang_reshape)
                
            coef_real = (coef_predict_mag*cos(coef_predict_ang)).reshape(-1)
            coef_imag = (coef_predict_mag*sin(coef_predict_ang)).reshape(-1)
    return coef_real, coef_imag

def set_training_params(training_params=None, training_conf=None):
    if training_conf is None:
        if training_params is None:
            training_params = {
                "processing_system": "bedroom",
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
                   test_size,
                   model_input_size,
                   output_sig_set,
                   use_premultiply,
                   nyfr,
                   dictionary,
                   training_params,
                   system_params,
                   mode,
                   premultiply_file_path):
    premultiply_sig_set = []
    premultiply_sig_set_train = np.zeros((train_size, model_input_size))
    premultiply_sig_set_test = np.zeros((test_size, model_input_size))
    complex_premultiply_sig_set_train = []
    complex_premultiply_sig_set_test = []
    for ij, output_signal in enumerate(output_sig_set):
        if ( use_premultiply ):
            premultiply_signal = np.copy(output_signal)
        else:
            if ( training_params['pre_omp'] ):
                original_recovery = system_params['recovery']
                system_params['recovery'] = 'c_omp'
                nyfr.set_system_params(system_params=system_params)
                premultiply_signal = nyfr.recover_signal(training_params['pre_multiply']*dictionary, output_signal)
                system_params['recovery'] = original_recovery
                nyfr.set_system_params(system_params=system_params)
            else:
                pseudo = np.linalg.pinv( training_params['pre_multiply'] * dictionary)
                premultiply_signal = np.dot(pseudo,output_signal)

            premultiply_sig_set.append(premultiply_signal)

        if ( ij < train_size ):
            if ( mode == 'real'):
                premultiply_sig_set_train[ij] = np.copy(premultiply_signal.real)
            elif ( mode == 'imag'):
                premultiply_sig_set_train[ij] = np.copy(premultiply_signal.imag)
            elif ( mode == 'mag'):
                premultiply_sig_set_train[ij] = np.abs(premultiply_signal)
            elif ( mode == 'ang'):
                premultiply_sig_set_train[ij] = np.angle(premultiply_signal)
            elif ( mode == 'complex'):
                premultiply_sig_concat = np.concatenate((premultiply_signal.real, premultiply_signal.imag))
                complex_premultiply_sig_set_train.append(premultiply_sig_concat)
        else:
            if ( mode == 'real'):
                premultiply_sig_set_test[ij - train_size] = np.copy(premultiply_signal.real)
            elif ( mode == 'imag'):
                premultiply_sig_set_test[ij - train_size] = np.copy(premultiply_signal.imag)
            elif ( mode == 'mag'):
                premultiply_sig_set_test[ij - train_size] = np.abs(premultiply_signal)
            elif ( mode == 'ang'):
                premultiply_sig_set_test[ij - train_size] = np.angle(premultiply_signal)
            elif ( mode == 'complex'):
                premultiply_sig_concat = np.concatenate((premultiply_signal.real, premultiply_signal.imag))
                complex_premultiply_sig_set_test.append(premultiply_sig_concat)

    if ( not use_premultiply and training_params['save_premultiply'] ):
        premultiply_sig_set_array = np.array(premultiply_sig_set)
        np.save(premultiply_file_path, premultiply_sig_set_array)

    if ( mode == 'complex'):
        premultiply_sig_set_train = np.array(complex_premultiply_sig_set_train)
        premultiply_sig_set_test = np.array(complex_premultiply_sig_set_test)

    return premultiply_sig_set_train, premultiply_sig_set_test

def create_model(mlp_model_file_path,
                 output_file_path,
                 recovery_log_file_path,
                 model_log_file_path,
                 model_input_size,
                 training_params,
                 premultiply_sig_set_train,
                 premultiply_sig_set_test,
                 output_sig_set_train,
                 output_sig_set_test,
                 nyfr):
    if ( os.path.isfile( mlp_model_file_path )):
        mlp_model = tf.keras.models.load_model(mlp_model_file_path)
    else:
        mlp_model = keras.Sequential()
        mlp_model.add(keras.Input(shape=(model_input_size,)))
        mlp_model.add(layers.Reshape((nyfr.get_Zones(), nyfr.get_K_band()), input_shape=(model_input_size,)))
        mlp_model.add(layers.Conv1D(filters=nyfr.get_Zones(),
                                    kernel_size=nyfr.get_K_band(),
                                    padding='same',
                                    input_shape=(nyfr.get_Zones(),nyfr.get_K_band()),
                                    # activity_regularizer=regularizers.l1(0.001),
                                    name="mlp_model_layer_1"))
        mlp_model.add(layers.Flatten())
        # mlp_model.add(layers.Dense(4*model_input_size, name="mlp_model_layer_2"))
        # mlp_model.add(layers.Dense(model_input_size, activity_regularizer=regularizers.l1(0.01), name="mlp_model_layer_2"))
        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_3"))
        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_4"))
        mlp_model.add(layers.Dense(nyfr.get_Zones(),
                                    activation='softmax',
                                #    activity_regularizer=regularizers.l2(0.001),
                                    name="mlp_model_out"))
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

def create_model_outputs(input_file_path, fft_file_path, active_zones_file_path, zones, training_params, mode):
    input_sig_set_total = np.load(input_file_path)
    num_input_sigs_total = input_sig_set_total.shape[0]
    input_sigs_not_used = num_input_sigs_total - training_params['total_num_sigs']
    _, input_sig_set = np.vsplit(input_sig_set_total, [input_sigs_not_used])
    active_zones = np.zeros(zones, dtype="float64")
    fft_sig_list = []
    active_zones_sig_list = []
    recovery_mode = ""

    for i, input_sig in enumerate(input_sig_set):
        input_sig_fft = fft(input_sig)
        fft_sig_list.append(input_sig_fft)
        input_zones = np.array_split(input_sig_fft, zones)
        non_zero_in_zones = [np.any(zone != 0) for zone in input_zones]
        for i, has_non_zero in enumerate(non_zero_in_zones):
            if has_non_zero:
                active_zones[i] = 1
        active_zones_sig_list.append(active_zones)

    fft_sig_set = np.array(fft_sig_list)
    if ( training_params['save_fft_file'] ):
        if ( not os.path.exists(fft_file_path) ):
            np.save(fft_file_path, fft_sig_set)
    
    active_zones_sig_set = np.array(active_zones_sig_list)
    if ( training_params['save_active_zones_file'] ):
        if ( not os.path.exists(active_zones_file_path) ):
            np.save(active_zones_file_path, active_zones_sig_set)

    output_sig_set = None
    recovery_mode = None
    if training_params['use_fft']:
        output_sig_list = []
        for i, fft_sig in enumerate(fft_sig_set):
            if (mode == 'real'):
                output_sig_list.append(fft_sig.real)
                recovery_mode = "real_imag"
            elif (mode == 'imag'):
                output_sig_list.append(fft_sig.real)
                recovery_mode = "real_imag"
            elif (mode == 'mag'):
                output_sig_list.append(np.abs(fft_sig))
                recovery_mode = "mag_ang"
            elif (mode == 'ang'):
                output_sig_list.append(np.angle(fft_sig))
                recovery_mode = "mag_ang"
            elif (mode == 'complex'):
                output_sig_list.append(np.concatenate((fft_sig.real, fft_sig.imag)))
                recovery_mode = "complex"
        output_sig_set = np.array(output_sig_list)
    elif training_params['use_active_zones']:
        output_sig_set = active_zones_sig_set
        recovery_mode = "active_zones"

    model_input_size = input_sig_set.shape[1]
    num_input_sigs = input_sig_set.shape[0]
    return output_sig_set, model_input_size, num_input_sigs, recovery_mode

def create_mlp1_models(NYFR_test_harness, training_params=None, training_conf=None):
    training_params = set_training_params(training_params=training_params, training_conf=training_conf)
    if training_params['use_fft'] and training_params['use_active_zones']:
        training_params['use_fft'] = False
    nyfr = NYFR_test_harness.get_nyfr()
    if nyfr is None:
        print("NYFR must be initialized")

    directories = NYFR_test_harness.get_directories()
    if directories is None:
        print("NYFR Test Harness must be initialized")
    files = NYFR_test_harness.get_filenames()
    recovery_params = nyfr.get_recovery_params()
    dictionary_params = nyfr.get_dictionary_params()
    system_params = nyfr.get_system_params()
    dictionary_file_base_dir = directories['dictionary'][dictionary_params['version']]
    dictionary_file_sub_dirs = get_all_sub_dirs(dictionary_file_base_dir)
    input_file_paths = get_all_file_paths(directories['input'])

    for mode in training_params['modes']:
        for input_file_path in input_file_paths:
            noise_level, phase_shift, file_name = get_file_sub_dirs(input_file_path)

            fft_file_sub_dir = directories['fft'] + noise_level + "\\" + phase_shift + "\\"
            fft_file_path = os.path.join(fft_file_sub_dir, file_name)
            fft_sig_set = None
            if ( os.path.isfile(fft_file_path) ):
                if fft_sig_set.shape[0] == training_params['total_num_sigs']:
                    fft_sig_set = np.load(fft_file_path)

            active_zones_sub_dir = directories['active_zones'] + noise_level + "\\" + phase_shift + "\\"
            active_zones_file_path = os.path.join(active_zones_sub_dir, file_name)
            active_zones_sig_set = None
            if ( os.path.isfile(active_zones_file_path) ):
                if active_zones_sig_set.shape[0] != training_params['total_num_sigs']:
                    active_zones_sig_set = np.load(active_zones_file_path)

            output_sig_set = None
            if training_params['use_active_zones']:
                if active_zones_sig_set is None:
                    output_sig_set, model_input_size, num_input_sigs, recovery_mode = create_model_outputs(input_file_path,
                                                                                                           fft_file_path,
                                                                                                           active_zones_file_path,
                                                                                                           nyfr.get_Zones(),
                                                                                                           training_params,
                                                                                                           mode)
                else:
                    output_sig_set = active_zones_sig_set
            elif training_params['use_fft']:
                if fft_sig_set is None:
                    output_sig_set, model_input_size, num_input_sigs, recovery_mode = create_model_outputs(input_file_path,
                                                                                                           fft_file_path,
                                                                                                           active_zones_file_path,
                                                                                                           nyfr.get_Zones(),
                                                                                                           training_params,
                                                                                                           mode)
                else: output_sig_set = fft_sig_set

            train_size = int(num_input_sigs * training_params['train_test_split_percentage'])
            test_size = num_input_sigs - train_size

            output_sig_set_train, output_sig_set_test = np.vsplit(output_sig_set, [train_size])
            output_sig_set = None
        
            output_sub_path = directories['output'] + noise_level + "\\" + phase_shift + "\\"
            mlp_model_sub_path = directories['mlp_models'][dictionary_params['version']][mode] + noise_level + "\\" + phase_shift + "\\"
            model_log_file_path = directories['mlp_models'][dictionary_params['version']][mode] + files['mlp_models']['log'][training_params['processing_systems']]
            premultiply_sub_path = directories['premultiply'][dictionary_params['version']] + noise_level + "\\" + phase_shift + "\\"
            if recovery_mode == "active_zones":
                recovery_log_file_path = directories['recovery'][dictionary_params['version']][recovery_params['type']] + "\\" + recovery_mode + "\\" + \
                        files['recovery'][recovery_mode][training_params['processing_system']]
            else:
                recovery_log_file_path = directories['recovery'][dictionary_params['version']][recovery_params['type']] + "\\" + recovery_mode + "\\" + \
                        files['recovery'][recovery_mode][mode][training_params['processing_system']]
            output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
            premultiply_file_sub_dirs = get_all_sub_dirs(premultiply_sub_path)
            mlp_model_file_sub_dirs = get_all_sub_dirs(mlp_model_sub_path)
            for index, sub_dir in enumerate(output_file_sub_dirs):
                output_file_path = os.path.join(sub_dir, file_name)
                found_string_in_file = False
                premultiply_file_path = os.path.join(premultiply_file_sub_dirs[index], file_name)
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
                        dictionary_file_path = dictionary_file_sub_dirs[index] + "\\" + files['dictionary']['name']
                        dictionary = np.load(dictionary_file_path)
                        output_sig_set_total = np.load(output_file_path)
                        num_input_sigs_total = output_sig_set_total.shape[0]
                        output_sigs_not_used = num_input_sigs_total - training_params['total_num_sigs']
                        _, output_sig_set = np.vsplit(output_sig_set_total, [output_sigs_not_used])
                        del output_sig_set_total

                    premultiply_sig_set_train, premultiply_sig_set_test = set_test_train(train_size,                                    
                                                                                         test_size,
                                                                                         model_input_size,
                                                                                         output_sig_set,
                                                                                         use_premultiply,
                                                                                         nyfr,
                                                                                         dictionary,
                                                                                         training_params,
                                                                                         system_params,
                                                                                         mode,
                                                                                         premultiply_file_path)

                    # input_file_name_without_extension = os.path.splitext(input_file_name)[0]
                    # mlp_model_file_path_ind = os.path.join(mlp_model_file_sub_dirs[index], input_file_name_without_extension + ".keras" )
                    mlp_model_file_path = os.path.join(mlp_model_file_sub_dirs[index], files['mlp_models']['name'])
                    create_model(mlp_model_file_path,
                                 output_file_path,
                                 recovery_log_file_path,
                                 model_log_file_path,
                                 model_input_size,
                                 training_params,
                                 premultiply_sig_set_train,
                                 premultiply_sig_set_test,
                                 output_sig_set_train,
                                 output_sig_set_test,
                                 nyfr)