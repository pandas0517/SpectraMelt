'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.Analog import Analog
    from spectramelt.NYFR import NYFR
    from spectramelt.Recovery import Recovery
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift
    import numpy as np
    import time
    import atexit

    load_dotenv()
    logger = get_logger(Path(__file__).stem)
    
    show_analog_signals = False
    show_wbf_signals = False
    show_LO_signals = False
    show_mixed_signals = False
    show_lpf_signals = True
    show_conditioned_signals = True
    show_ADC_signals = True
    show_recovered_signals = False
    show_premultiply_signals = True
    
    analog_1 = Analog(config_file_path=Path(getenv('INPUT_CONF')))
    analog_sig_1 = analog_1.create_analog()
    
    real_time_1 = analog_sig_1.time
    real_freq_1 = analog_sig_1.frequency
    total_time_1 = analog_sig_1.total_time
    sim_freq_1 = analog_1.get_time_params().get('sim_freq')

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    input_signal_1_wave_params = input_signal_1.get_wave_params()
    input_signal_1_waves = input_signal_1_wave_params.get('waves')
    input_signal_1_wave_params["v_ref_range"] = (-2.5, 2.5)
    input_signal_1.set_wave_params(input_signal_1_wave_params)

    real_input_1 = input_signal_1.create_input_signal(real_time=real_time_1)
    real_input_time_1 = real_input_1.input_signal
    real_input_freq_1 = fftshift(np.abs(fft(real_input_1.input_signal))) / (sim_freq_1*total_time_1)
    real_input_phase_1 = fftshift(np.angle(fft(real_input_1.input_signal)))

    if show_analog_signals:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 2 rows, 3 columns
        axes[0].plot(real_time_1, real_input_time_1)
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(real_freq_1, real_input_freq_1)
        axes[1].set_title("Frequency - Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        axes[2].plot(real_freq_1, real_input_phase_1)
        axes[2].set_title("Frequency - Phase (Ideal)")
        axes[2].set_xlim(-300000, 300000)
        fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
   
    nyfr_1 = NYFR(config_file_path=Path(getenv('NYFR_CONF')))
    start = time.time()
    nyfr_signal_1 = nyfr_1.create_output_signal(real_input_time_1, real_time_1,
                                                return_internal=True)
    end = time.time()
    logger.info(f"NYFR execution time with file config: {end - start:.6f} seconds")

    wbf_params_1 = nyfr_1.get_wbf_params()
    wbf_samp_freq_1 = wbf_params_1.get('cutoff_freq')
    wbf_time_1 = nyfr_signal_1.wbf_signal.time
    wbf_freq_1 = nyfr_signal_1.wbf_signal.freq
    wbf_signal_1 = nyfr_signal_1.wbf_signal.wbf_signal.filtered
    wbf_sig_freq_1 = fftshift(np.abs(fft(wbf_signal_1))) / (sim_freq_1*total_time_1)

    lo_signal_1 = nyfr_signal_1.lo_signal.lo
    lo_freq_1 = fftshift(np.abs(fft(lo_signal_1))) / (sim_freq_1*total_time_1)

    pulse_signal_1 = nyfr_signal_1.pulse_signal.pulses
    pulse_freq_1 = fftshift(np.abs(fft(pulse_signal_1))) / (sim_freq_1*total_time_1)

    mixed_signal_1 = nyfr_signal_1.mixed_signal.mixed
    mixed_freq_1 = fftshift(np.abs(fft(mixed_signal_1))) / (sim_freq_1*total_time_1)

    lpf_params_1 = nyfr_1.get_lpf_params()
    lpf_signal_1 = nyfr_signal_1.lpf_signal.filtered
    lpf_freq_1 = fftshift(np.abs(fft(lpf_signal_1))) / (sim_freq_1*total_time_1)

    lpf_cond_sigs_1 = nyfr_signal_1.adc_signal.conditioned
    lpf_cond_sig_1 = lpf_cond_sigs_1.signal
    lpf_cond_time_1 = lpf_cond_sigs_1.time
    lpf_cond_freq_1 = lpf_cond_sigs_1.freq
    lpf_cond_total_time_1 = lpf_cond_sigs_1.total_time 
    lpf_cond_sig_freq_1 = fftshift(np.abs(fft(lpf_cond_sig_1))) / (sim_freq_1*lpf_cond_total_time_1)

    bits_nyfr_1 = nyfr_1.get_adc_params().get('num_bits')
    sh_output_nyfr_1 = nyfr_signal_1.adc_signal.sample_hold.output_signal
    mid_times_nyfr_1 = nyfr_signal_1.adc_signal.quantized.mid_times
    quantized_nyfr_1 = nyfr_signal_1.adc_signal.quantized.quantized_values
    samp_freq_nyfr_1 = nyfr_signal_1.adc_signal.quantized.sampled_frequency
    quant_freq_nyfr_1 = fftshift(np.abs(fft(quantized_nyfr_1))) / len(samp_freq_nyfr_1)

    lo_1_params = nyfr_1.get_lo_params()
    lo_freq = lo_1_params.get('freq')
    adc_1_params = nyfr_1.get_adc_params()
    samp_freq = nyfr_signal_1.adc_signal.quantized.sampled_frequency

    nyfr_lpf_waves = []
    nyfr_adc_waves = []

    for input_wave in input_signal_1_waves:
        input_freq = input_wave['freq']

        base_fold = input_freq - lo_freq * round(input_freq / lo_freq)
        for folded_freq in (np.abs(base_fold), -np.abs(base_fold)):

            real_freq_idx = np.abs(real_freq_1 - folded_freq).argmin()
            samp_freq_idx = np.abs(samp_freq - folded_freq).argmin()

            nyfr_lpf_wave = input_wave.copy()
            nyfr_lpf_wave['amp'] = lpf_freq_1[real_freq_idx]
            nyfr_lpf_wave['freq'] = real_freq_1[real_freq_idx]
            nyfr_lpf_waves.append(nyfr_lpf_wave)

            nyfr_adc_wave = input_wave.copy()
            nyfr_adc_wave['amp'] = quant_freq_nyfr_1[samp_freq_idx]
            nyfr_adc_wave['freq'] = samp_freq[samp_freq_idx]
            nyfr_adc_waves.append(nyfr_adc_wave)

    if show_wbf_signals:
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(real_time_1, wbf_signal_1)
        axes[0].set_title(f"Time (Ideal)\nUsing filter mode {wbf_params_1['mode']}")
        axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(real_freq_1, wbf_sig_freq_1)
        axes[1].set_title("Frequency Magnitude (Ideal)")
        axes[1].set_xlim(-300000, 300000)
        fig.suptitle("NYFR Wide Band Filtered Signals ")
        fig.tight_layout()
        plt.show()

    if show_LO_signals:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))
        axes[0].plot(real_time_1, lo_signal_1)
        axes[0].plot(real_time_1, pulse_signal_1)
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(-0.0000125, 0.0000125)
        axes[1].plot(real_freq_1, lo_freq_1)
        axes[1].set_title("LO Frequency (File)")
        axes[1].set_xlim(-120000, 120000)
        axes[2].plot(real_freq_1, pulse_freq_1)
        axes[2].set_title("Pulse Frequency (File)")
        axes[2].set_xlim(-120000, 120000)
        fig.suptitle("LO and Pulse Signals")
        fig.tight_layout()
        plt.show()

    if show_mixed_signals:
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(real_time_1, mixed_signal_1, label="Mixed")
        axes[0].plot(real_time_1, wbf_signal_1, label="Wideband Filtered")
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(-0.0001, 0.0001)
        axes[0].legend()
        axes[1].plot(real_freq_1, mixed_freq_1)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(-120000, 120000)
        fig.suptitle("NYFR Mixed Signals")
        fig.tight_layout()
        plt.show()
    
    if show_lpf_signals:
        nyfr_freqs = np.array([w['freq'] for w in nyfr_lpf_waves])
        nyfr_amps  = np.array([w['amp']  for w in nyfr_lpf_waves])
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(real_time_1, lpf_signal_1)
        axes[0].set_title(f"Time (File)\nUsing filter mode {lpf_params_1['mode']}")
        axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(real_freq_1, lpf_freq_1)
        axes[1].set_title("Frequency Magnitude (File)")
        axes[1].set_xlim(-75000, 75000)
        axes[1].scatter(
            nyfr_freqs,
            nyfr_amps,
            marker="x",
            color='red',
            s=100,
            label="NYFR Folded Waves"
        )
        # fig.suptitle("NYFR Low Pass Filtered Signals")
        axes[1].legend()
        fig.tight_layout()
        plt.show()
 
    if show_conditioned_signals:
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
    
    if show_ADC_signals:
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

    start = time.time()
    nyfr_dict_1 = nyfr_1.create_dictionary(nyfr_signal_1.lo_phase_mod_mid, wbf_time_1)
    end = time.time()
    logger.info(f"NYFR dictionary creation time with file config: {end - start:.6f} seconds")  

    dictionary_1 = nyfr_dict_1.dictionary
    start = time.time()
    pinv_1 = np.linalg.pinv(100*dictionary_1)
    premultiply_1 = pinv_1 @ quantized_nyfr_1
    end = time.time()
    logger.info(f"Premultiplication Time: {end - start:.6f} seconds")

    recovery_1 = Recovery(config_file_path=Path(getenv('RECOVERY_CONF')))
    recovery_method_1 = "spgl1" 
    recovery_1.set_recovery_method(recovery_method_1)

    start = time.time()    
    recovered_freq_1 = recovery_1.recover_signal(quantized_nyfr_1, nyfr_dict_1.dictionary)
    end = time.time()
    logger.info(f"NYFR SPGL-1 recovery time with file config: {end - start:.6f} seconds")  

    recovered_sig_freq_1 = fftshift(np.abs(fft(recovered_freq_1))) / (wbf_samp_freq_1*lpf_cond_total_time_1)
    recovered_signal_1 = ifft(recovered_freq_1)

    if show_recovered_signals:
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
    
    if show_premultiply_signals:
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