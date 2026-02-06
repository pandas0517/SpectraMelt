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
    from spectramelt.NFWBS import NFWBS
    from spectramelt.Recovery import Recovery
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift
    import numpy as np
    import time
    import atexit

    load_dotenv()
    logger = get_logger(Path(__file__).stem)
    
    display_analog_signals = False
    display_wbf_signals = False
    display_LO_1_signals = False
    display_mixed_1_signals = False
    display_lpf_1_signals = False
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
    analog_1_time_params["time_range"] = (-2, 2)
    # analog_1_time_params["sim_freq"] = 100000
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
    
    if display_analog_signals:
        tmin, tmax = -50e-6, 50e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot = real_input_time_1[mask_t]
        
        fmin, fmax = -300_000, 300_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot = real_input_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot)
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("Simulated Analog Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
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

    wbf_time_1 = wbf_signal_1.time
    wbf_freq_1 = wbf_signal_1.freq

    if display_wbf_signals:
        tmin, tmax = -50e-6, 50e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot = wbf_signal_1.wbf_signal.filtered[mask_t]
        
        fmin, fmax = -300_000, 300_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot = wbf_sig_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot)
        axes[0].set_title(f"Time (File)\nUsing filter mode {wbf_params_1['mode']}")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NYFR Stage Wide Band Filtered Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    if display_LO_1_signals:
        tmin, tmax = -15e-6, 15e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot_1 = lo_1_signal_1.lo[mask_t]
        y_plot_2 = pulse_1_signal_1.pulses[mask_t]
        
        fmin, fmax = -120_000, 120_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot_1 = lo_1_freq_1[mask_f]
        Y_plot_2 = pulse_1_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 3, figsize=(8,4))
        axes[0].plot(t_plot, y_plot_1)
        axes[0].plot(t_plot, y_plot_2)
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot_1)
        axes[1].set_title("LO Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        axes[2].plot(f_plot, Y_plot_2)
        axes[2].set_title("Pulse Frequency (File)")
        axes[2].set_xlim(fmin, fmax)
        fig.suptitle("First Stage LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    if display_mixed_1_signals:
        tmin, tmax = -50e-6, 50e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot_1 = wbf_signal_1.wbf_signal.filtered[mask_t]
        y_plot_2 = mixed_1_signal_1.mixed[mask_t]
        
        fmin, fmax = -120_000, 120_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot_1 = mixed_1_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot_2, label="Mixed Signal")
        axes[0].plot(t_plot, y_plot_1, label="Wide-band Filtered")
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(tmin, tmax)
        axes[0].legend()
        axes[1].plot(f_plot, Y_plot_1)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NYFR Stage Mixed Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
    if display_lpf_1_signals:
        tmin, tmax = -200e-6, 200e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot = lpf_1_signal_1.filtered[mask_t]
        
        fmin, fmax = -100_000, 100_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot = lpf_1_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot)
        axes[0].set_title(f"Time (File)\nUsing filter mode {lpf_1_params_1['mode']}")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NYFR Stage Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
 
    if display_LO_2_signals:
        tmin, tmax = -2, 2
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot_1 = lo_2_signal_1.lo[mask_t]
        y_plot_2 = pulse_2_signal_1.pulses[mask_t]
        
        fmin_1, fmax_1 = -5, 5
        mask_f_1 = (real_freq_1 >= fmin_1) & (real_freq_1 <= fmax_1)

        f_plot_1 = real_freq_1[mask_f_1]
        Y_plot_1 = lo_2_freq_1[mask_f_1]
        
        fmin_2, fmax_2 = -300_000, 300_000
        mask_f_2 = (real_freq_1 >= fmin_2) & (real_freq_1 <= fmax_2)
        f_plot_2 = real_freq_1[mask_f_2]       
        Y_plot_2 = pulse_2_freq_1[mask_f_2]
        
        fig, axes = plt.subplots(1, 3, figsize=(8,4))
        axes[0].plot(t_plot, y_plot_1)
        axes[0].plot(t_plot, y_plot_2)
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot_1, Y_plot_1)
        axes[1].set_title("LO Frequency (File)")
        axes[1].set_xlim(fmin_1, fmax_1)
        axes[2].plot(f_plot_2, Y_plot_2)
        axes[2].set_title("Pulse Frequency (File)")
        axes[2].set_xlim(fmin_2, fmax_2)
        fig.suptitle("Second Stage LO and Pulse Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        
    if display_wavelet_signals:
        tmin, tmax = -100e-6, 200e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot = wavelet_signal_1.wavelet_train.real[mask_t]
        
        fmin, fmax = -120_000, 120_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot = wavelet_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot)
        axes[0].set_title("Time (File)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (File)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NFWBS Wavelet Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    if display_mixed_2_signals:
        tmin, tmax = -500e-6, 500e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot_1 = lpf_1_signal_1.filtered[mask_t]
        y_plot_2 = mixed_2_signal_1.mixed[mask_t]
        
        fmin, fmax = -120_000, 120_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot_1 = mixed_2_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 3, figsize=(8,4))
        axes[0].plot(t_plot, y_plot_1, label="Low-pass Filtered")
        axes[0].set_title("Time (Low-pass Filtered)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(t_plot, y_plot_2.real, label="Mixed Signal")
        axes[1].set_title("Time (Mixed Signal)")
        axes[1].set_xlim(tmin, tmax)
        axes[2].plot(f_plot, Y_plot_1)
        axes[2].set_title("Frequency (Mixed Signal)")
        axes[2].set_xlim(fmin, fmax)
        fig.suptitle("NFWBS Mixed Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
    if display_lpf_2_signals:
        tmin, tmax = -200e-6, 200e-6
        mask_t = (real_time_1 >= tmin) & (real_time_1 <= tmax)

        t_plot = real_time_1[mask_t]
        y_plot = lpf_2_signal_1.filtered[mask_t]
        
        fmin, fmax = -100_000, 100_000
        mask_f = (real_freq_1 >= fmin) & (real_freq_1 <= fmax)

        f_plot = real_freq_1[mask_f]
        Y_plot = lpf_2_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot.real)
        axes[0].set_title("Time (Ideal)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (Ideal)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NFWBS Stage Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
 
    if display_conditioned_signals:
        tmin, tmax = -200e-6, 200e-6
        mask_t = (lpf_cond_time_1 >= tmin) & (lpf_cond_time_1 <= tmax)

        t_plot = lpf_cond_time_1[mask_t]
        y_plot = lpf_cond_sig_1[mask_t]
        
        fmin, fmax = -40_000, 40_000
        mask_f = (lpf_cond_freq_1 >= fmin) & (lpf_cond_freq_1 <= fmax)

        f_plot = lpf_cond_freq_1[mask_f]
        Y_plot = lpf_cond_sig_freq_1[mask_f]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(t_plot, y_plot)
        axes[0].set_title(f"Time (Ideal)")
        axes[0].set_xlim(tmin, tmax)
        axes[1].plot(f_plot, Y_plot)
        axes[1].set_title("Frequency (Ideal)")
        axes[1].set_xlim(fmin, fmax)
        fig.suptitle("NFWBS Conditioned Low Pass Filtered Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
    if display_ADC_signals:
        fig, axes = plt.subplots(1, 3, figsize=(8,4))
        axes[0].plot(lpf_cond_time_1, lpf_cond_sig_1)
        axes[0].plot(lpf_cond_time_1, sh_output_nfwbs_1)
        axes[0].set_title("Sample and Hold - Time (Ideal)")
        axes[0].set_xlim(0, 0.00015)
        axes[1].step(mid_times_nfwbs_1, quantized_nfwbs_1, color='green', where='mid')
        axes[1].set_xlim(-.0001, 0.0003)
        axes[1].set_title(f"{bits_nyfr_1}-bit quantizer - Time (Ideal)")
        axes[2].plot(samp_freq_nfwbs_1, quant_freq_nfwbs_1)
        axes[2].set_ylim(0, 0.00025)
        axes[2].set_title(f"{bits_nyfr_1}-bit quantizer - Frequency (Ideal)")
        fig.suptitle("NFWBS ADC Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
        
    # dictionary_1 = nyfr_1.get_nyfr_dict()
    # pinv_1 = np.linalg.pinv(100*dictionary_1)
    # premultiply_1 = pinv_1 @ quantized_nyfr_1
    
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

    if display_recovered_signals:
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        axes[0].plot(real_freq_1, real_input_freq_1)
        axes[0].set_title("Frequency (File)")
        axes[0].set_ylim(0, 1)
        axes[0].set_xlim(-50000, 50000)
        #axes[0].set_xlim(-0.0002, 0.0002)
        axes[1].plot(wbf_freq_1, recovered_sig_freq_1)
        axes[1].set_title(f"Frequency (File)\nRecovery Method: {recovery_method_1}")
        axes[1].set_ylim(0, 1)
        axes[1].set_xlim(-50000, 50000)
        fig.suptitle("NFWBS Recovered Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    
    if display_premultiply_signals:
        premult_mag_1 = fftshift(np.abs(fft(premultiply_1))) / len(premultiply_1)
        fig, axes = plt.subplots(1, 2, figsize=(8,4))  # 2 rows, 2 columns
        axes[0].plot(real_freq_1, wbf_sig_freq_1)
        axes[0].set_title("Wide-band Filtered Frequency Magnitude (Ideal)")
        axes[0].set_xlim(-160000, 160000)
        axes[1].plot(wbf_freq_1, premult_mag_1)
        axes[1].set_title("Premultiply Frequency Magnitude (Ideal)")
        # axes[0].set_ylim(0, 1)
        # axes[0].set_xlim(-50000, 50000)
        # axes[1].set_ylim(0, 1)
        # axes[1].set_xlim(-50000, 50000)
        fig.suptitle("NFWBS Premultiply Signals")
        fig.tight_layout()
        plt.show()
        plt.close(fig)
             
    atexit.register(logger.info, "Completed Test\n")