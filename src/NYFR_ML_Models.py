import keras
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras import layers
from keras import losses
import tensorflow as tf
from numpy import sin, cos
import numpy as np


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

def create_mlp1_models(system_params):
    # modes = [ 'imag', 'mag', 'ang', 'complex']
    # modes = [ 'real', 'imag', 'mag', 'ang', 'complex']
    # modes = [ 'real']
    # modes = [ 'mag', 'ang' ]
    time_file_path = "test_sets\\System_Config_1_Internal\\System\\Not_Sampled\\time.npy"
    t_test = np.load(time_file_path)
    K_band = round(t_test.size*(system_params['spacing']*system_params['adc_clock_freq']))
    Zones = int(t_test.size/K_band)
    del t_test
    modes = [ 'mag' ]
    pre_omp = False
    total_num_sigs = 40000
    dictionary_file_name = "dictionary.npy"
    mlp_model_file_name = "mlp_model_file.keras"
    # mlp_model_file_name = "input_list.txt"
    # model_log_file_name = "input_list_Bedroom_PC.txt"
    train_test_split_percentage = 0.7
    learning_rate = 0.00001
    num_epochs = 200
    batch_sz = 128
    recovery_dic_type = 'original'
    # processing_systems = 'daddo'
    processing_systems = 'bedroom'
    # mlp_model_file_name = "input_list.txt"
    model_log_file_name = {
        'bedroom':"input_list_Bedroom_PC.txt",
        'daddo':"input_list_Daddo_PC.txt"
    }
    recovery_log_file_name = {
        'mag':{
            'rec_mode': 'mag_ang',
            'bedroom': "recovery_list_bedroom_mag.txt",
            'daddo': "recovery_list_daddo_mag.txt"
        },
        'ang':{
            'rec_mode': 'mag_ang',
            'bedroom': "recovery_list_bedroom_ang.txt",
            'daddo': "recovery_list_daddo_ang.txt"
        },
    }
    directory_list = {
        'input': "test_sets\\System_Config_1_Inputs\\",
        'fft': "test_sets\\System_Config_1_Model_Inputs\\FFT\\",
        'train': "test_sets\\System_Config_1_Model_Inputs\\train\\",
        'test': "test_sets\\System_Config_1_Model_Inputs\\test\\",
        'output': "test_sets\\System_Config_1_Outputs\\",
        'mlp_models': {
            'enhanced': {
                'real': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\real\\",
                'imag': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\imag\\",
                'mag': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\mag\\",
                'ang': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\ang\\",
                'complex': "F:\\test_sets\\System_Config_1_MLP_Models\\enhanced\\complex\\"
            },
            'original': {
                'real': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\real\\",
                'imag': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\imag\\",
                'mag': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\mag\\",
                'ang': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\ang\\",
                'complex': "F:\\test_sets\\System_Config_1_MLP_Models\\original\\complex\\"
            }
        },
        'dictionary': {
            'enhanced': "test_sets\\System_Config_1_Internal\\Dictionary\\enhanced\\",
            'original': "test_sets\\System_Config_1_Internal\\Dictionary\\original\\"
        },
        'premultiply': {
            'enhanced': "Y:\\School_Stuff\\System_Config_1_PreMultiply\\enhanced\\",
            'original': "Y:\\School_Stuff\\System_Config_1_PreMultiply\\original\\"
        },
        'recovery': {
            'enhanced': {
                'c_omp': "test_sets\\System_Config_1_Recovery\\OMP_Custom\\enhanced\\",
                'o_omp': "test_sets\\System_Config_1_Recovery\\OMP\\enhanced\\",
                'mlp1': "test_sets\\System_Config_1_Recovery\\MLP1\\enhanced\\",
                'spgl1': "test_sets\\System_Config_1_Recovery\\SPGL\\enhanced\\"
            },
            'original': {
                'c_omp': "test_sets\\System_Config_1_Recovery\\OMP_Custom\\original\\",
                'o_omp': "test_sets\\System_Config_1_Recovery\\OMP\\original\\",
                'mlp1': "test_sets\\System_Config_1_Recovery\\MLP1\\original\\",
                'spgl1': "test_sets\\System_Config_1_Recovery\\SPGL\\original\\"
            }
        }
    }

    dictionary_file_base_dir = directory_list['dictionary'][recovery_dic_type]
    dictionary_file_sub_dirs = get_all_sub_dirs(dictionary_file_base_dir)
    input_file_paths = get_all_file_paths(directory_list['input'])
    use_fft_file = False
    save_fft_file = True
    use_premultiply = False
    save_premultiply = True

    for mode in modes:
        # train_input_file_paths = get_all_file_paths(directory_list['train'])
        # test_input_file_paths = get_all_file_paths(directory_list['test'])

        for id, input_set_file in enumerate(input_file_paths):
            input_path = Path(input_set_file)
            input_path_len = len(input_path.parts)
            input_file_name = input_path.parts[input_path_len - 1]
            input_phase_shift = input_path.parts[input_path_len - 2]
            input_noise_level = input_path.parts[input_path_len - 3]
            fft_file_sub_dir = directory_list['fft'] + input_noise_level + "\\" + input_phase_shift + "\\"
            fft_file_path = os.path.join(fft_file_sub_dir, input_file_name)
            
            if ( os.path.isfile(fft_file_path) ):
                input_sig_set = np.load(fft_file_path)
                use_fft_file = True
            else:
            # for i, train_input_file in enumerate(train_input_file_paths):
                use_fft_file = False
                input_sig_set_total = np.load(input_set_file)
                num_input_sigs_total = input_sig_set_total.shape[0]
                input_sigs_not_used = num_input_sigs_total - total_num_sigs
                # input_sig_set_split = np.vsplit(input_sig_set_total, 2)
                input_sig_set_split = np.vsplit(input_sig_set_total, [input_sigs_not_used])
                input_sig_set = np.copy(input_sig_set_split[1])
                del input_sig_set_split
                del input_sig_set_total
            # test_size = input_sig_set_test.shape[0]
            # train_size = input_sig_set_train.shape[0]
            fft_input_sig_set = []
            complex_input_sig_set = []

            for i, input_sig in enumerate(input_sig_set):
                if ( use_fft_file ):
                    input_sig_fft = input_sig
                else:
                    input_sig_fft = fft(input_sig)
                    if ( save_fft_file ):
                        fft_input_sig_set.append(input_sig_fft)
                # input_sig_fft_real = input_sig_fft.real
                # input_sig_fft_imag = input_sig_fft.imag
                if (mode == 'real'):
                    input_sig_set[i] = np.copy(input_sig_fft.real)
                elif (mode == 'imag'):
                    input_sig_set[i] = np.copy(input_sig_fft.imag)
                elif (mode == 'mag'):
                    input_sig_set[i] = np.abs(input_sig_fft)
                elif (mode == 'ang'):
                    input_sig_set[i] = np.angle(input_sig_fft)
                elif (mode == 'complex'):
                    input_sig_concat = np.concatenate((input_sig_fft.real, input_sig_fft.imag))
                    complex_input_sig_set.append(input_sig_concat)

            if ( not use_fft_file and save_fft_file ):
                fft_input_set = np.array(fft_input_sig_set)
                if ( not os.path.exists(fft_file_path) ):
                    np.save(fft_file_path, fft_input_set)
                del fft_input_set
                del fft_input_sig_set

            if ( mode == 'complex'):
                input_sig_set = np.array(complex_input_sig_set)
                del complex_input_sig_set
            elif ( use_fft_file ):
                input_sig_set = input_sig_set.astype(np.float64)

            model_input_size = input_sig_set.shape[1]
            num_input_sigs = input_sig_set.shape[0]
            train_size = int(num_input_sigs * train_test_split_percentage)
            test_size = num_input_sigs - train_size

            input_sig_set_train_test = np.vsplit(input_sig_set, [train_size])
            del input_sig_set
        
            output_sub_path = directory_list['output'] + input_noise_level + "\\" + input_phase_shift + "\\"
            mlp_model_sub_path = directory_list['mlp_models'][recovery_dic_type][mode] + input_noise_level + "\\" + input_phase_shift + "\\"
            model_log_file_path = directory_list['mlp_models'][recovery_dic_type][mode] + model_log_file_name[processing_systems]
            premultiply_sub_path = directory_list['premultiply'][recovery_dic_type] + input_noise_level + "\\" + input_phase_shift + "\\"
            recovery_log_file_path = directory_list['recovery'][recovery_dic_type][system_params['recovery']] \
                    + recovery_log_file_name[mode]['rec_mode'] + "\\" + recovery_log_file_name[mode][processing_systems]
            output_file_sub_dirs = get_all_sub_dirs(output_sub_path)
            premultiply_file_sub_dirs = get_all_sub_dirs(premultiply_sub_path)
            mlp_model_file_sub_dirs = get_all_sub_dirs(mlp_model_sub_path)
            # for index, sub_dir in enumerate(premultiply_file_sub_dirs):
            for index, sub_dir in enumerate(output_file_sub_dirs):
                output_file_path = os.path.join(sub_dir, input_file_name)
                found_string_in_file = False
                premultiply_file_path = os.path.join(premultiply_file_sub_dirs[index], input_file_name)
                with open(model_log_file_path, "r") as model_log:
                    for line in model_log:
                        # if premultiply_file_path in line:
                        if output_file_path in line:
                            found_string_in_file = True
                            break

                if not found_string_in_file:  
                    #del model_log
                    # del line
                    if ( os.path.isfile(premultiply_file_path) ):
                        output_sig_set = np.load(premultiply_file_path)

                        use_premultiply = True
                    else:
                        use_premultiply = False
                        dictionary_file_path = dictionary_file_sub_dirs[index] + "\\" + dictionary_file_name
                        dictionary = np.load(dictionary_file_path)
                        output_sig_set_total = np.load(output_file_path)
                        num_input_sigs_total = output_sig_set_total.shape[0]
                        output_sigs_not_used = num_input_sigs_total - total_num_sigs
                        output_sig_set_split = np.vsplit(output_sig_set_total, [output_sigs_not_used])
                        output_sig_set = output_sig_set_split[1]
                        del output_sig_set_total
                        del output_sig_set_split

                    premultiply_sig_set = []
                    premultiply_sig_set_train = np.zeros((train_size, model_input_size))
                    premultiply_sig_set_test = np.zeros((test_size, model_input_size))
                    complex_premultiply_sig_set_train = []
                    complex_premultiply_sig_set_test = []
                    # start_time = time.perf_counter()
                    for ij, output_signal in enumerate(output_sig_set):
                        if ( use_premultiply ):
                            premultiply_signal = np.copy(output_signal)
                        else:
                            if ( pre_omp ):
                                original_recovery = system_params['recovery']
                                system_params['recovery'] = 'c_omp'
                                premultiply_signal = recover_signal(0.2*dictionary, output_signal, system_params, mode)
                                system_params['recovery'] = original_recovery
                            else:
                                pseudo = np.linalg.pinv( 0.01 * dictionary)
                                premultiply_signal = np.dot(pseudo,output_signal)

                            premultiply_sig_set.append(premultiply_signal)

                        # premultiply_signal_real = premultiply_signal.real
                        # premultiply_signal_imag = premultiply_signal.imag

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
                        # premultiply_sig_flattened[ij] = np.concatenate((premultiply_signal_real, premultiply_signal_imag))
                        pass

                    if ( not use_premultiply and save_premultiply ):
                        premultiply_sig_set_array = np.array(premultiply_sig_set)
                        np.save(premultiply_file_path, premultiply_sig_set_array)
                        del premultiply_sig_set_array
                        del premultiply_sig_set

                    if ( mode == 'complex'):
                        premultiply_sig_set_train = np.array(complex_premultiply_sig_set_train)
                        premultiply_sig_set_test = np.array(complex_premultiply_sig_set_test)
                        del complex_premultiply_sig_set_train
                        del complex_premultiply_sig_set_test

                    # input_file_name_without_extension = os.path.splitext(input_file_name)[0]
                    # mlp_model_file_path_ind = os.path.join(mlp_model_file_sub_dirs[index], input_file_name_without_extension + ".keras" )
                    mlp_model_file_path = os.path.join(mlp_model_file_sub_dirs[index], mlp_model_file_name)
                    if ( os.path.isfile( mlp_model_file_path )):
                        mlp_model = tf.keras.models.load_model(mlp_model_file_path)
                    else:
                        # if ( mode == 'ang' ):
                        #     loss_type = 'mean_squared_error'
                        # else:
                            # loss_type = 'mean_squared_error'
                            # loss_type = 'mean_absolute_error'
                        # loss_type = 'mean_squared_logarithmic_error'
                        # loss_type = 'Huber'
                        # loss_type = 'mean_absolute_error'
                        # loss_type = 'LogCosh'
                        loss_type = 'root_mean_squared_error'
                        mlp_model = keras.Sequential()
                        mlp_model.add(keras.Input(shape=(model_input_size,)))
                        mlp_model.add(layers.Reshape((Zones, K_band), input_shape=(model_input_size,)))
                        mlp_model.add(layers.Conv1D(filters=K_band,
                                                    kernel_size=10,
                                                    padding='same',
                                                    input_shape=(Zones,K_band),
                                                    # activity_regularizer=regularizers.l1(0.001),
                                                    name="mlp_model_layer_1"))
                        mlp_model.add(layers.Flatten())
                        # mlp_model.add(layers.Dense(4*model_input_size, name="mlp_model_layer_2"))
                        # mlp_model.add(layers.Dense(model_input_size, activity_regularizer=regularizers.l1(0.01), name="mlp_model_layer_2"))
                        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_3"))
                        # mlp_model.add(layers.Dense(model_input_size, name="mlp_model_layer_4"))
                        mlp_model.add(layers.Dense(model_input_size,
                                                   activation='linear',
                                                #    activity_regularizer=regularizers.l2(0.001),
                                                   name="mlp_model_out"))
                        # mlp_model.add(layers.Activation('relu'))
                        mlp_opt = keras.optimizers.Adam(learning_rate=learning_rate)
                        # mlp_model.compile(optimizer=mlp_opt, loss=loss_type)
                        mlp_model.compile(optimizer=mlp_opt, loss=root_mean_squared_error)

                    early_stopping = EarlyStopping(monitor='val_loss',
                                                   min_delta=0.1,
                                                   patience=4,
                                                   verbose=1,
                                                   start_from_epoch=5,
                                                   restore_best_weights=True)
                    mlp_model.fit(premultiply_sig_set_train, input_sig_set_train_test[0],
                                    epochs=num_epochs,
                                    batch_size=batch_sz,
                                    shuffle=True,
                                    validation_data=(premultiply_sig_set_test, input_sig_set_train_test[1]),
                                    callbacks=[early_stopping])
                    mlp_model.save(mlp_model_file_path, overwrite=True)
                    # mlp_model.save(mlp_model_file_path_ind, overwrite=True)

                    with open(model_log_file_path, "a") as model_log:
                        model_log.write(output_file_path + "\n")
                    with open(recovery_log_file_path, "a") as recovery_log:
                        recovery_log.write(output_file_path + "\n")
                    reset_tensforflow_session()