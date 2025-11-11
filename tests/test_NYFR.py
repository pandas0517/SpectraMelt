'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger, load_config_from_json
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.NYFR import NYFR
    from spectramelt.Recovery import Recovery
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift
    import numpy as np
    import time
    import atexit

    load_dotenv()
    logger = get_logger(Path(__file__).stem)
    
    display_analog_signals = True
    display_wbf_signals = False
    display_LO_signals = False
    display_mixed_signals = False
    display_lpf_signals = False
    display_conditioned_signals = True
    display_ADC_signals = True
    display_recovered_signals = False
    display_premultiply_signals = True
    
    input_conf_1 = load_config_from_json(Path(getenv('INPUT_CONF')))
    input_signal_1 = InputSignal(input_conf_1)
    real_time_1 = input_signal_1.get_analog_time()
    real_input_1 = input_signal_1.get_input_signal()
    real_freq_1 = input_signal_1.get_analog_frequency()
    total_time_1 = input_signal_1.get_analog_signal_params().get('total_time')
    sim_freq_1 = input_signal_1.get_analog_signals().get('sim_freq')
    real_input_freq_1 = np.fft.fftshift(np.abs(fft(real_input_1))) / (sim_freq_1*total_time_1)
    
    input_signal_2 = InputSignal()
    real_time_2 = input_signal_2.get_analog_time()
    real_input_2 = input_signal_2.get_input_signal()
    real_freq_2 = input_signal_2.get_analog_frequency()
    total_time_2 = input_signal_2.get_analog_signal_params().get('total_time')
    sim_freq_2 = input_signal_2.get_analog_signals().get('sim_freq')
    real_input_freq_2 = np.fft.fftshift(np.abs(fft(real_input_2))) / (sim_freq_2*total_time_2)
    
    nyfr_config_1 = load_config_from_json(Path(getenv('NYFR_CONF')))
    start = time.time()
    nyfr_1 = NYFR(real_input_1, real_time_1, nyfr_config_1)
    end = time.time()
    logger.info(f"NYFR Execution time with file config: {end - start:.6f} seconds")
    
    wbf_signal_1 = nyfr_1.get_wbf_signal()
    wbf_params_1 = nyfr_1.get_wbf_params()
    wbf_samp_freq_1 = wbf_params_1.get('cutoff_freq')
    wbf_sig_freq_1 = fftshift(np.abs(fft(wbf_signal_1))) / (sim_freq_1*total_time_1)
    lo_signal_1 = nyfr_1.get_lo_signal()
    lo_freq_1 = fftshift(np.abs(fft(lo_signal_1))) / (sim_freq_1*total_time_1)
    pulse_signal_1 = nyfr_1.get_pulse_signal()
    pulse_freq_1 = fftshift(np.abs(fft(pulse_signal_1))) / (sim_freq_1*total_time_1)
    mixed_signal_1 = nyfr_1.get_mixed_signal()
    mixed_freq_1 = fftshift(np.abs(fft(mixed_signal_1))) / (sim_freq_1*total_time_1)
    lpf_params_1 = nyfr_1.get_lpf_params()
    lpf_signal_1 = nyfr_1.get_lpf_signal()
    lpf_freq_1 = fftshift(np.abs(fft(lpf_signal_1))) / (sim_freq_1*total_time_1)
    lpf_cond_sigs_1 = nyfr_1.get_conditioned_signals()
    lpf_cond_sig_1 = lpf_cond_sigs_1.get('signal')
    lpf_cond_time_1 = lpf_cond_sigs_1.get('time')
    lpf_cond_freq_1 = lpf_cond_sigs_1.get('freq')
    lpf_cond_total_time_1 = lpf_cond_sigs_1.get('total_time') 
    lpf_cond_sig_freq_1 = fftshift(np.abs(fft(lpf_cond_sig_1))) / (sim_freq_1*lpf_cond_total_time_1)
    bits_nyfr_1 = nyfr_1.get_adc_params().get('num_bits')
    sh_output_nyfr_1 = nyfr_1.get_sh_signals().get('output_signal')
    mid_times_nyfr_1 = nyfr_1.get_output_signals().get('mid_times')
    quantized_nyfr_1 = nyfr_1.get_output_signals().get('quantized_values')
    samp_freq_nyfr_1 = nyfr_1.get_output_signals().get('sampled_frequency')
    quant_freq_nyfr_1 = fftshift(np.abs(fft(quantized_nyfr_1))) / len(samp_freq_nyfr_1)
    dictionary_1 = nyfr_1.get_nyfr_dict()
    pinv_1 = np.linalg.pinv(100*dictionary_1)
    premultiply_1 = pinv_1 @ quantized_nyfr_1
    wbf_time_1 = nyfr_1.get_wbf_time()
    wbf_freq_1 = nyfr_1.get_wbf_freq()
    recovery_config_path = Path(getenv('RECOVERY_CONF'))
    recovery_1 = Recovery(quantized_nyfr_1, dictionary_1, config_file_path=recovery_config_path)
    recovery_params_1 = recovery_1.get_recovery_params()
    recovery_method_1 = recovery_params_1.get('method')
    recovered_freq_1 = recovery_1.get_recovered_coefs()
    recovered_sig_freq_1 = fftshift(np.abs(fft(recovered_freq_1))) / (wbf_samp_freq_1*lpf_cond_total_time_1)
    recovered_signal_1 = ifft(recovered_freq_1)

    start = time.time()
    premultiply_signal_1 = np.dot(np.linalg.pinv(dictionary_1), quantized_nyfr_1)
    end = time.time()
    logger.info(f"Premultiplication Time: {end - start:.6f} seconds")

    start = time.time()
    nyfr_2 = NYFR(real_input_2, real_time_2)
    end = time.time()
    logger.info(f"NYFR Execution time with default config: {end - start:.6f} seconds")
    
    wbf_signal_2 = nyfr_2.get_wbf_signal()
    wbf_params_2 = nyfr_2.get_wbf_params()
    wbf_samp_freq_2 = wbf_params_2.get('cutoff_freq')
    wbf_sig_freq_2 = fftshift(np.abs(fft(wbf_signal_2))) / (sim_freq_2*total_time_2)
    lo_signal_2 = nyfr_2.get_lo_signal()
    lo_freq_2 = fftshift(np.abs(fft(lo_signal_2))) / (sim_freq_2*total_time_2)
    pulse_signal_2 = nyfr_2.get_pulse_signal()
    pulse_freq_2 = fftshift(np.abs(fft(pulse_signal_2))) / (sim_freq_2*total_time_2)
    mixed_signal_2 = nyfr_2.get_mixed_signal()
    mixed_freq_2 = fftshift(np.abs(fft(mixed_signal_2))) / (sim_freq_2*total_time_2)
    lpf_params_2 = nyfr_2.get_lpf_params()
    lpf_signal_2 = nyfr_2.get_lpf_signal()
    lpf_freq_2 = fftshift(np.abs(fft(lpf_signal_2))) / (sim_freq_2*total_time_2)  
    lpf_cond_sigs_2 = nyfr_2.get_conditioned_signals()
    lpf_cond_sig_2 = lpf_cond_sigs_2.get('signal')
    lpf_cond_time_2 = lpf_cond_sigs_2.get('time')
    lpf_cond_freq_2 = lpf_cond_sigs_2.get('freq') 
    lpf_cond_total_time_2 = lpf_cond_sigs_2.get('total_time') 
    lpf_cond_sig_freq_2 = fftshift(np.abs(fft(lpf_cond_sig_2))) / (sim_freq_2*lpf_cond_total_time_2)
    bits_nyfr_2 = nyfr_2.get_adc_params().get('num_bits')
    sh_output_nyfr_2 = nyfr_2.get_sh_signals().get('output_signal')
    mid_times_nyfr_2 = nyfr_2.get_output_signals().get('mid_times')
    quantized_nyfr_2 = nyfr_2.get_output_signals().get('quantized_values')
    samp_freq_nyfr_2 = nyfr_2.get_output_signals().get('sampled_frequency')
    quant_freq_nyfr_2 = fftshift(np.abs(fft(quantized_nyfr_2))) / len(samp_freq_nyfr_2)
    dictionary_2 = nyfr_2.get_nyfr_dict()
    pinv_2 = np.linalg.pinv(dictionary_2)
    premultiply_2 = pinv_2 @ quantized_nyfr_2
    wbf_time_2 = nyfr_2.get_wbf_time()
    wbf_freq_2 = nyfr_2.get_wbf_freq()

    recovery_2 = Recovery(quantized_nyfr_2, dictionary_2)
    recovery_params_2 = recovery_2.get_recovery_params()
    recovery_method_2 = recovery_params_2.get('method')
    recovered_freq_2 = recovery_2.get_recovered_coefs()
    recovered_sig_freq_2 = fftshift(np.abs(fft(recovered_freq_2))) / (wbf_samp_freq_2*lpf_cond_total_time_2)
    recovered_signal_2 = ifft(recovered_freq_2)
    
    if display_analog_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, real_input_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, real_input_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-160000, 160000)
        axes[1,0].plot(real_time_2, real_input_2)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, real_input_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1600, 1600)
        fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()

    if display_wbf_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, wbf_signal_1)
        axes[0,0].set_title(f"Time (File)\nUsing filter mode {wbf_params_1['mode']}")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, wbf_sig_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-160000, 160000)
        axes[1,0].plot(real_time_2, wbf_signal_2)
        axes[1,0].set_title(f"Time (File)\nUsing filter mode {wbf_params_2['mode']}")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, wbf_sig_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1600, 1600)
        fig.suptitle("NYFR Wide Band Filtered Signals ")
        fig.tight_layout()
        plt.show()

    if display_LO_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0,0].plot(real_time_1, lo_signal_1)
        axes[0,0].plot(real_time_1, pulse_signal_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lo_freq_1)
        axes[0,1].set_title("LO Frequency (File)")
        axes[0,1].set_xlim(-12000, 12000)
        axes[0,2].plot(real_freq_1, pulse_freq_1)
        axes[0,2].set_title("Pulse Frequency (File)")
        axes[0,2].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, lo_signal_2)
        axes[1,0].plot(real_time_2, pulse_signal_2)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.05)
        axes[1,1].plot(real_freq_2, lo_freq_2)
        axes[1,1].set_title("LO Frequency (Default)")
        axes[1,1].set_xlim(-120, 120)
        axes[1,2].plot(real_freq_2, pulse_freq_2)
        axes[1,2].set_title("Pulse Frequency (File)")
        axes[1,2].set_xlim(-1200, 1200)
        fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()

    if display_mixed_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, wbf_signal_1)
        axes[0,0].plot(real_time_1, mixed_signal_1)
        axes[0,0].set_title("Time (File)")
        axes[0,0].set_xlim(-0.0005, 0.0005)
        axes[0,1].plot(real_freq_1, mixed_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-120000, 120000)
        axes[1,0].plot(real_time_2, wbf_signal_2)
        axes[1,0].plot(real_time_2, mixed_signal_2)
        axes[1,0].set_title("Time (Default)")
        axes[1,0].set_xlim(0, 0.1)
        axes[1,1].plot(real_freq_2, mixed_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-1200, 1200)
        fig.suptitle("NYFR Mixed Signals")
        fig.tight_layout()
        plt.show()
    
    if display_lpf_signals:
        fig, axes = plt.subplots(2, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(real_time_1, lpf_signal_1)
        axes[0,0].set_title(f"Time (File)\nUsing filter mode {lpf_params_1['mode']}")
        #axes[0,0].set_xlim(-0.0002, 0.0002)
        axes[0,1].plot(real_freq_1, lpf_freq_1)
        axes[0,1].set_title("Frequency (File)")
        axes[0,1].set_xlim(-13000, 13000)
        axes[1,0].plot(real_time_2, lpf_signal_2)
        axes[1,0].set_title(f"Time (File)\nUsing filter mode {lpf_params_2['mode']}")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].plot(real_freq_2, lpf_freq_2)
        axes[1,1].set_title("Frequency (Default)")
        axes[1,1].set_xlim(-130, 130)
        fig.suptitle("NYFR Low Pass Filtered Signals")
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
        fig.suptitle("NYFR Conditioned Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
    
    if display_ADC_signals:
        fig, axes = plt.subplots(2, 3, figsize=(8,4))  # 2 rows, 2 columns
        axes[0,0].plot(lpf_cond_time_1, lpf_cond_sig_1)
        axes[0,0].plot(lpf_cond_time_1, sh_output_nyfr_1)
        axes[0,0].set_title("Sample and Hold - Time (File)")
        axes[0,0].set_xlim(-0.0003, 0.0003)
        axes[0,1].step(mid_times_nyfr_1, quantized_nyfr_1, color='green', where='mid')
        axes[0,1].set_title(f"{bits_nyfr_1}-bit quantizer - Time (File)")
        axes[0,2].plot(samp_freq_nyfr_1, quant_freq_nyfr_1)
        axes[0,2].set_ylim(0, 0.2)
        axes[0,2].set_title(f"{bits_nyfr_1}-bit quantizer - Frequency (File)")
        axes[1,0].plot(lpf_cond_time_2, lpf_cond_sig_2)
        axes[1,0].plot(lpf_cond_time_2, sh_output_nyfr_2)
        axes[1,0].set_title("Sample and Hold - Time (Default)")
        axes[1,0].set_xlim(0, 0.04)
        axes[1,1].step(mid_times_nyfr_2, quantized_nyfr_2, color='green', where='mid')
        axes[1,1].set_title(f"{bits_nyfr_2}-bit quantizer - Time (Default)")
        axes[1,2].plot(samp_freq_nyfr_2, quant_freq_nyfr_2)
        axes[1,2].set_ylim(0, 0.2)
        axes[1,2].set_title(f"{bits_nyfr_1}-bit quantizer - Frequency (File)")
        fig.suptitle("NYFR ADC Signals")
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
        fig.suptitle("NYFR Recovered Signals")
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
        fig.suptitle("Premultiply Signals")
        fig.tight_layout()
        plt.show()
             
    atexit.register(logger.info, "Completed Test\n")