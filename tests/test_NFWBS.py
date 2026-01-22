'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger, load_config_from_json
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.Analog import Analog
    from spectramelt.NFWBS import NFWBS
    from spectramelt.Recovery import Recovery
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift
    import numpy as np
    import time
    import atexit

    load_dotenv()
    logger = get_logger(Path(__file__).stem)
    
    display_analog_signals = True
    display_wbf_signals = True
    display_LO_1_signals = True
    display_mixed_1_signals = True
    display_lpf_1_signals = True
    display_LO_2_signals = True
    display_wavelet_signals = True
    display_mixed_2_signals = True
    display_lpf_2_signals = True
    display_conditioned_signals = True
    display_ADC_signals = True
    display_recovered_signals = False
    display_premultiply_signals = False
    
    use_gpu = False
    use_device_message = "CPU"
    if use_gpu:
        use_device_message = "GPU"
    
    nfwbs_1 = NFWBS(config_file_path=Path(getenv('NFWBS_CONF')))
    nfwbs_1_adc_params = nfwbs_1.get_adc_params()
    
    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    analog_1_time_params = analog_1.get_time_params()
    analog_1_time_params["adc_samp_freq"] = nfwbs_1_adc_params.get('adc_samp_freq')
    analog_1.set_time_params(analog_1_time_params)
    analog_sig_1 = analog_1.create_analog()
    
    real_time_1 = analog_sig_1.time
    real_freq_1 = analog_sig_1.frequency
    total_time_1 = analog_sig_1.total_time
    sim_freq_1 = analog_1.get_time_params().get('sim_freq')

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    real_input_1 = input_signal_1.create_input_signal(real_time=real_time_1)
    real_input_time_1 = real_input_1.input_signal
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1.input_signal))) / (sim_freq_1*total_time_1)
    
    nfwbs_2 = NFWBS()
    nfwbs_2_adc_params = nfwbs_2.get_adc_params()  

    analog_2 = Analog()
    analog_2_time_params = analog_2.get_time_params()
    analog_2_time_params["time_range"] = (0, 5)
    analog_2_time_params["sim_freq"] = 100000
    analog_2_time_params["adc_samp_freq"] = nfwbs_2_adc_params.get('adc_samp_freq')
    analog_2.set_time_params(analog_2_time_params)
    analog_sig_2 = analog_2.create_analog()

    real_time_2 = analog_sig_2.time
    real_freq_2 = analog_sig_2.frequency
    total_time_2 = analog_sig_2.total_time
    sim_freq_2 = analog_2.get_time_params().get('sim_freq')

    input_signal_2 = InputSignal()
    real_input_2 = input_signal_2.create_input_signal(real_time=real_time_2)
    real_input_time_2 = real_input_2.input_signal
    real_input_freq_2 = fftshift(np.abs(fft(real_input_2.input_signal))) / (sim_freq_2*total_time_2)
    
    if display_analog_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, real_input_time_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, real_input_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-160000, 160000)
        axes[1,0].plot(real_time_2, real_input_time_2)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, real_input_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1600, 1600)
        fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
    
    start = time.time()
    nfwbs_1_signals = nfwbs_1.create_output_signal(input_signal=real_input_1.input_signal,
                                                   real_time=real_time_1,
                                                   return_internal=True,
                                                   use_gpu=use_gpu)
    end = time.time()
    logger.info(f"NFWBS Execution time with file config using {use_device_message}: {end - start:.6f} seconds")
    
    wbf_signal_1 = nfwbs_1_signals.wbf_signal
    wbf_params_1 = nfwbs_1.get_wbf_params()
    wbf_samp_freq_1 = wbf_params_1.get('cutoff_freq')
    wbf_sig_freq_1 = fftshift(np.abs(fft(wbf_signal_1.wbf_signal.filtered))) / (sim_freq_1*total_time_1)
    lo_1_signal_1 = nfwbs_1_signals.lo_1_signal
    lo_1_freq_1 = fftshift(np.abs(fft(lo_1_signal_1.lo))) / (sim_freq_1*total_time_1)
    pulse_1_signal_1 = nfwbs_1_signals.pulse_1_signal
    pulse_1_freq_1 = fftshift(np.abs(fft(pulse_1_signal_1.pulses))) / (sim_freq_1*total_time_1)
    mixed_1_signal_1 = nfwbs_1_signals.mixed_1_signal
    mixed_1_freq_1 = fftshift(np.abs(fft(mixed_1_signal_1.mixed))) / (sim_freq_1*total_time_1)
    lpf_1_params_1 = nfwbs_1.get_lpf_1_params()
    lpf_1_signal_1 = nfwbs_1_signals.lpf_1_signal
    lpf_1_freq_1 = fftshift(np.abs(fft(lpf_1_signal_1.filtered))) / (sim_freq_1*total_time_1)
    lo_2_signal_1 = nfwbs_1_signals.lo_2_signal
    lo_2_freq_1 = fftshift(np.abs(fft(lo_2_signal_1.lo))) / (sim_freq_1*total_time_1)
    pulse_2_signal_1 = nfwbs_1_signals.pulse_2_signal
    pulse_2_freq_1 = fftshift(np.abs(fft(pulse_2_signal_1.pulses))) / (sim_freq_1*total_time_1)
    wavelet_signal_1 = nfwbs_1_signals.wavelet_signal
    wavelet_freq_1 = fftshift(np.abs(fft(wavelet_signal_1.wavelet_train))) / (sim_freq_1*total_time_1)
    mixed_2_signal_1 = nfwbs_1_signals.mixed_2_signal
    mixed_2_freq_1 = fftshift(np.abs(fft(mixed_2_signal_1.mixed))) / (sim_freq_1*total_time_1)
    lpf_2_params_1 = nfwbs_1.get_lpf_1_params()
    lpf_2_signal_1 = nfwbs_1_signals.lpf_2_signal
    lpf_2_freq_1 = fftshift(np.abs(fft(lpf_2_signal_1.filtered))) / (sim_freq_1*total_time_1)
    lpf_cond_sig_1 = nfwbs_1_signals.adc_signal.conditioned.signal
    lpf_cond_time_1 = nfwbs_1_signals.adc_signal.conditioned.time
    lpf_cond_freq_1 = nfwbs_1_signals.adc_signal.conditioned.freq
    lpf_cond_total_time_1 = nfwbs_1_signals.adc_signal.conditioned.total_time
    lpf_cond_sig_freq_1 = fftshift(np.abs(fft(lpf_cond_sig_1))) / (sim_freq_1*lpf_cond_total_time_1)
    bits_nyfr_1 = nfwbs_1_adc_params.get('num_bits')
    sh_output_nfwbs_1 = nfwbs_1_signals.adc_signal.sample_hold.output_signal
    mid_times_nfwbs_1 = nfwbs_1_signals.adc_signal.quantized.mid_times
    quantized_nfwbs_1 = nfwbs_1_signals.adc_signal.quantized.quantized_values
    samp_freq_nfwbs_1 = nfwbs_1_signals.adc_signal.quantized.sampled_frequency
    quant_freq_nfwbs_1 = fftshift(np.abs(fft(quantized_nfwbs_1))) / len(samp_freq_nfwbs_1)
    # dictionary_1 = nyfr_1.get_nyfr_dict()
    # pinv_1 = np.linalg.pinv(100*dictionary_1)
    # premultiply_1 = pinv_1 @ quantized_nyfr_1
    wbf_time_1 = wbf_signal_1.time
    wbf_freq_1 = wbf_signal_1.freq
    # recovery_config_path = Path(getenv('RECOVERY_CONF'))
    # recovery_1 = Recovery(quantized_nyfr_1, dictionary_1, config_file_path=recovery_config_path)
    # recovery_params_1 = recovery_1.get_recovery_params()
    # recovery_method_1 = recovery_params_1.get('method')
    # recovered_freq_1 = recovery_1.get_recovered_coefs()
    # recovered_sig_freq_1 = fftshift(np.abs(fft(recovered_freq_1))) / (wbf_samp_freq_1*lpf_cond_total_time_1)
    # recovered_signal_1 = ifft(recovered_freq_1)

    # start = time.time()
    # premultiply_signal_1 = np.dot(np.linalg.pinv(dictionary_1), quantized_nyfr_1)
    # end = time.time()
    # logger.info(f"Premultiplication Time: {end - start:.6f} seconds")
    

    
    start = time.time()
    nfwbs_2_signals = nfwbs_2.create_output_signal(input_signal=real_input_2.input_signal,
                                                   real_time=real_time_2,
                                                   return_internal=True,
                                                   use_gpu=use_gpu)
    end = time.time()
    logger.info(f"NFWBS Execution time with default config using {use_device_message}: {end - start:.6f} seconds")
    
    wbf_signal_2 = nfwbs_2_signals.wbf_signal
    wbf_params_2 = nfwbs_2.get_wbf_params()
    wbf_samp_freq_2 = wbf_params_2.get('cutoff_freq')
    wbf_sig_freq_2 = fftshift(np.abs(fft(wbf_signal_2.wbf_signal.filtered))) / (sim_freq_2*total_time_2)
    lo_1_signal_2 = nfwbs_2_signals.lo_1_signal
    lo_1_freq_2 = fftshift(np.abs(fft(lo_1_signal_2.lo))) / (sim_freq_2*total_time_2)
    pulse_1_signal_2 = nfwbs_2_signals.pulse_1_signal
    pulse_1_freq_2 = fftshift(np.abs(fft(pulse_1_signal_2.pulses))) / (sim_freq_2*total_time_2)
    mixed_1_signal_2 = nfwbs_2_signals.mixed_1_signal
    mixed_1_freq_2 = fftshift(np.abs(fft(mixed_1_signal_2.mixed))) / (sim_freq_2*total_time_2)
    lpf_1_params_2 = nfwbs_2.get_lpf_1_params()
    lpf_1_signal_2 = nfwbs_2_signals.lpf_1_signal
    lpf_1_freq_2 = fftshift(np.abs(fft(lpf_1_signal_2.filtered))) / (sim_freq_2*total_time_2)
    lo_2_signal_2 = nfwbs_2_signals.lo_2_signal
    lo_2_freq_2 = fftshift(np.abs(fft(lo_2_signal_2.lo))) / (sim_freq_2*total_time_2)
    pulse_2_signal_2 = nfwbs_2_signals.pulse_2_signal
    pulse_2_freq_2 = fftshift(np.abs(fft(pulse_2_signal_2.pulses))) / (sim_freq_2*total_time_2)
    wavelet_signal_2 = nfwbs_2_signals.wavelet_signal
    wavelet_freq_2 = fftshift(np.abs(fft(wavelet_signal_2.wavelet_train))) / (sim_freq_2*total_time_2)
    mixed_2_signal_2 = nfwbs_2_signals.mixed_2_signal
    mixed_2_freq_2 = fftshift(np.abs(fft(mixed_2_signal_2.mixed))) / (sim_freq_2*total_time_2)
    lpf_2_params_2 = nfwbs_2.get_lpf_2_params()
    lpf_2_signal_2 = nfwbs_2_signals.lpf_2_signal
    lpf_2_freq_2 = fftshift(np.abs(fft(lpf_2_signal_2.filtered))) / (sim_freq_2*total_time_2)
    lpf_cond_sig_2 = nfwbs_2_signals.adc_signal.conditioned.signal
    lpf_cond_time_2 = nfwbs_2_signals.adc_signal.conditioned.time
    lpf_cond_freq_2 = nfwbs_2_signals.adc_signal.conditioned.freq
    lpf_cond_total_time_2 = nfwbs_2_signals.adc_signal.conditioned.total_time
    lpf_cond_sig_freq_2 = fftshift(np.abs(fft(lpf_cond_sig_2))) / (sim_freq_2*lpf_cond_total_time_2)
    bits_nyfr_2 = nfwbs_2_adc_params.get('num_bits')
    sh_output_nfwbs_2 = nfwbs_2_signals.adc_signal.sample_hold.output_signal
    mid_times_nfwbs_2 = nfwbs_2_signals.adc_signal.quantized.mid_times
    quantized_nfwbs_2 = nfwbs_2_signals.adc_signal.quantized.quantized_values
    samp_freq_nfwbs_2 = nfwbs_2_signals.adc_signal.quantized.sampled_frequency
    quant_freq_nfwbs_2 = fftshift(np.abs(fft(quantized_nfwbs_2))) / len(samp_freq_nfwbs_2)
    # dictionary_1 = nyfr_1.get_nyfr_dict()
    # pinv_1 = np.linalg.pinv(100*dictionary_1)
    # premultiply_1 = pinv_1 @ quantized_nyfr_1
    wbf_time_2 = wbf_signal_2.time
    wbf_freq_2 = wbf_signal_2.freq
    # recovery_config_path = Path(getenv('RECOVERY_CONF'))
    # recovery_1 = Recovery(quantized_nyfr_1, dictionary_1, config_file_path=recovery_config_path)
    # recovery_params_1 = recovery_1.get_recovery_params()
    # recovery_method_1 = recovery_params_1.get('method')
    # recovered_freq_1 = recovery_1.get_recovered_coefs()
    # recovered_sig_freq_1 = fftshift(np.abs(fft(recovered_freq_1))) / (wbf_samp_freq_1*lpf_cond_total_time_1)
    # recovered_signal_1 = ifft(recovered_freq_1)

    # start = time.time()
    # premultiply_signal_1 = np.dot(np.linalg.pinv(dictionary_1), quantized_nyfr_1)
    # end = time.time()
    # logger.info(f"Premultiplication Time: {end - start:.6f} seconds")

    if display_wbf_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, wbf_signal_1.wbf_signal.filtered)
        axes[0,0].set_title(f"Time (File)\nUsing filter mode {wbf_params_1['mode']}")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, wbf_sig_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-160000, 160000)
        axes[1,0].plot(real_time_2, wbf_signal_2.wbf_signal.filtered)
        axes[1,0].set_title(f"Time (File)\nUsing filter mode {wbf_params_2['mode']}")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, wbf_sig_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1600, 1600)
        fig.suptitle("NYFR Stage Wide Band Filtered Signals ")
        fig.tight_layout()
        plt.show()

    if display_LO_1_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lo_1_signal_1.lo)
        axes[0,0].plot(real_time_1, pulse_1_signal_1.pulses)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lo_1_freq_1)
        axes[0,1].set_title("LO Frequency (File)")
        axes[0,1].set_xlim(-12000, 12000)
        axes[0,2].plot(real_freq_1, pulse_1_freq_1)
        axes[0,2].set_title("Pulse Frequency (File)")
        axes[0,2].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, lo_1_signal_2.lo)
        axes[1,0].plot(real_time_2, pulse_1_signal_2.pulses)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lo_1_freq_2)
        axes[1,1].set_title("LO Frequency (Default)")
        axes[1,1].set_xlim(-120, 120)
        axes[1,2].plot(real_freq_2, pulse_1_freq_2)
        axes[1,2].set_title("Pulse Frequency (File)")
        axes[1,2].set_xlim(-1200, 1200)
        fig.suptitle("First Stage LO and Pulse Signals")
        fig.tight_layout()
        plt.show()

    if display_mixed_1_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, wbf_signal_1.wbf_signal.filtered)
        axes[0,0].plot(real_time_1, mixed_1_signal_1.mixed)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0005, 0.0005)
        axes[0,1].plot(real_freq_1, mixed_1_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, wbf_signal_2.wbf_signal.filtered)
        axes[1,0].plot(real_time_2, mixed_1_signal_2.mixed)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.1)
        axes[1,1].plot(real_freq_2, mixed_1_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1200, 1200)
        fig.suptitle("NYFR Stage Mixed Signals")
        fig.tight_layout()
        plt.show()
    
    if display_lpf_1_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, lpf_1_signal_1.filtered)
        axes[0,0].set_title(f"Time (File)\nUsing filter mode {lpf_1_params_1['mode']}")
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lpf_1_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-13000, 13000)
        axes[1,0].plot(real_time_2, lpf_1_signal_2.filtered)
        axes[1,0].set_title(f"Time (File)\nUsing filter mode {lpf_1_params_2['mode']}")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, lpf_1_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-130, 130)
        fig.suptitle("NYFR Stage Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
 
    if display_LO_2_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lo_2_signal_1.lo)
        axes[0,0].plot(real_time_1, pulse_2_signal_1.pulses)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lo_2_freq_1)
        axes[0,1].set_title("LO Frequency (File)")
        axes[0,1].set_xlim(-12000, 12000)
        axes[0,2].plot(real_freq_1, pulse_2_freq_1)
        axes[0,2].set_title("Pulse Frequency (File)")
        axes[0,2].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, lo_2_signal_2.lo)
        axes[1,0].plot(real_time_2, pulse_2_signal_2.pulses)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lo_2_freq_2)
        axes[1,1].set_title("LO Frequency (Default)")
        axes[1,1].set_xlim(-120, 120)
        axes[1,2].plot(real_freq_2, pulse_2_freq_2)
        axes[1,2].set_title("Pulse Frequency (File)")
        axes[1,2].set_xlim(-1200, 1200)
        fig.suptitle("Second Stage LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
        
    if display_wavelet_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, wavelet_signal_1.wavelet_train.real)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0005, 0.0005)
        axes[0,1].plot(real_freq_1, wavelet_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, wavelet_signal_2.wavelet_train.real)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.1)
        axes[1,1].plot(real_freq_2, wavelet_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1200, 1200)
        fig.suptitle("NFWBS Wavelet Signals")
        fig.tight_layout()
        plt.show()

    if display_mixed_2_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, lpf_1_signal_1.filtered)
        axes[0,0].plot(real_time_1, mixed_2_signal_1.mixed)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0005, 0.0005)
        axes[0,1].plot(real_freq_1, mixed_2_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, lpf_1_signal_2.filtered)
        axes[1,0].plot(real_time_2, mixed_2_signal_2.mixed)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.1)
        axes[1,1].plot(real_freq_2, mixed_2_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1200, 1200)
        fig.suptitle("NFWBS Mixed Signals")
        fig.tight_layout()
        plt.show()
    
    if display_lpf_2_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, lpf_2_signal_1.filtered)
        axes[0,0].set_title(f"Time (File)\nUsing filter mode {lpf_2_params_1['mode']}")
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lpf_2_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-13000, 13000)
        axes[1,0].plot(real_time_2, lpf_2_signal_2.filtered)
        axes[1,0].set_title(f"Time (File)\nUsing filter mode {lpf_1_params_2['mode']}")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, lpf_2_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-130, 130)
        fig.suptitle("NFWBS Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
 
    if display_conditioned_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(lpf_cond_time_1, lpf_cond_sig_1)
        axes[0,0].set_title("Time (File)")
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(lpf_cond_freq_1, lpf_cond_sig_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_xlim(-10000, 10000)
        axes[1,0].plot(lpf_cond_time_2, lpf_cond_sig_2)
        #axes[1,0].set_title("Time (Default)")
        #axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(lpf_cond_freq_2, lpf_cond_sig_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-130, 130)
        fig.suptitle("NFWBS Conditioned Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
    
    if display_ADC_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(lpf_cond_time_1, lpf_cond_sig_1)
        axes[0,0].plot(lpf_cond_time_1, sh_output_nfwbs_1)
        axes[0,0].set_title("Sample and Hold - Time (File)")
        axes[0,0].set_xlim(-0.0003, 0.0003)
        axes[0,1].step(mid_times_nfwbs_1, quantized_nfwbs_1, color='green', where='mid')
        axes[0,1].set_title(f"{bits_nyfr_1}-bit quantizer - Time (File)")
        axes[0,2].plot(samp_freq_nfwbs_1, quant_freq_nfwbs_1)
        axes[0,2].set_ylim(0, 0.2)
        axes[0,2].set_title(f"{bits_nyfr_1}-bit quantizer - Frequency (File)")
        axes[1,0].plot(lpf_cond_time_2, lpf_cond_sig_2)
        axes[1,0].plot(lpf_cond_time_2, sh_output_nfwbs_2)
        axes[1,0].set_title("Sample and Hold - Time (Default)")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].step(mid_times_nfwbs_2, quantized_nfwbs_2, color='green', where='mid')
        axes[1,1].set_title(f"{bits_nyfr_2}-bit quantizer - Time (Default)")
        axes[1,2].plot(samp_freq_nfwbs_2, quant_freq_nfwbs_2)
        axes[1,2].set_ylim(0, 0.2)
        axes[1,2].set_title(f"{bits_nyfr_1}-bit quantizer - Frequency (File)")
        fig.suptitle("NFWBS ADC Signals")
        fig.tight_layout()
        plt.show()

    if display_recovered_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_freq_1, real_input_freq_1)
        axes[0,0].set_title("Frequency (File)")
        axes[0,0].set_ylim(0, 1)
        axes[0,0].set_xlim(-50000, 50000)
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(wbf_freq_1, recovered_sig_freq_1)
        axes[0,1].set_title(f"Frequency (File)\nRecovery Method: {recovery_method_1}")
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_xlim(-50000, 50000)
        axes[1,0].plot(real_freq_2, real_input_freq_2)
        axes[1,0].set_title(f"Time (Default)")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(wbf_freq_2, recovered_sig_freq_2)
        axes[1,1].set_title(f"Frequency (Default)\nRecovery Method: {recovery_method_2}")
        axes[1,1].set_xlim(-130, 130)
        fig.suptitle("NFWBS Recovered Signals")
        fig.tight_layout()
        plt.show()
    
    if display_premultiply_signals:
        premult_mag_1 = fftshift(np.abs(fft(premultiply_1))) / len(premultiply_1)
        premult_mag_2 = fftshift(np.abs(fft(premultiply_2))) / len(premultiply_2)
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_freq_1, wbf_sig_freq_1)
        axes[0,0].set_title("Frequency (File)")
        axes[0,0].set_xlim(-160000, 160000)
        axes[0,1].plot(wbf_freq_1, premult_mag_1)
        axes[0,1].set_title("Premultiply Frequency (File)")
        # axes[0].set_ylim(0, 1)
        # axes[0].set_xlim(-50000, 50000)
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[1,0].plot(real_freq_2, wbf_sig_freq_2)
        axes[1,0].set_title("Frequency (File)")
        axes[1,0].set_xlim(-1600, 1600)
        axes[1,1].plot(wbf_freq_2, premult_mag_2)
        axes[1,1].set_title("Premultiply Frequency (Default)")
        # axes[1].set_ylim(0, 1)
        # axes[1].set_xlim(-50000, 50000)
        fig.suptitle("NFWBS Premultiply Signals")
        fig.tight_layout()
        plt.show()
             
    atexit.register(logger.info, "Completed Test\n")