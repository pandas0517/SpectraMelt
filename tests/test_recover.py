'''
@author: pete
'''
if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import ifft, ifftshift

    wide_band_freq_file = Path("F:\\Dataset_Config_1\\Input_Config_Ideal_1\\NYFR_Config_1\\Outputs\Wideband\\wbf_time_freq.npz")

    with np.load(wide_band_freq_file) as time_freq:
        wideband_time = time_freq["time"]
        wideband_freq = time_freq["freq"]    

    wideband_time_file = Path("F:\\Dataset_Config_1\\Input_Config_Ideal_1\\NYFR_Config_1\\Outputs\\Wideband\\2_tone_recovery_time_signals.npy") 
    wideband_freq_file = Path("F:\\Dataset_Config_1\\Input_Config_Ideal_1\\NYFR_Config_1\\Outputs\\Wideband\\2_tone_recovery_freq_signals.npz")

    wideband_time_signals = np.load(wideband_time_file)

    with np.load(wideband_freq_file) as freq:
        wideband_mag_signals = freq["mag"]
        wideband_phase_signals = freq["ang"]
        wideband_real_signals = freq["real"]
        wideband_imag_signals = freq["imag"]
        wideband_complex_signals = wideband_real_signals + 1j * wideband_imag_signals

    recovery_time_file = Path("F:\\Dataset_Config_1\\Input_Config_Ideal_1\\NYFR_Config_1\\Recovery_Config_1\\Recovery\\2_tone_recovery_time_signals.npy")
    recovery_freq_file = Path("F:\\Dataset_Config_1\\Input_Config_Ideal_1\\NYFR_Config_1\\Recovery_Config_1\\Recovery\\2_tone_recovery_freq_signals.npz")

    recovery_time_signals = np.load(recovery_time_file)

    with np.load(recovery_freq_file) as freq:
        recovery_mag_signals = freq["mag"]
        recovery_real_imag_signals = freq["real_imag"]
        recovery_real_signals = np.hsplit(recovery_real_imag_signals, 2)[0]
        recovery_imag_signals = np.hsplit(recovery_real_imag_signals, 2)[1]
        recovery_complex_signals = recovery_real_signals + 1j * recovery_imag_signals

    idx = 1

    complex_rec_signal = np.real(ifft(ifftshift(recovery_complex_signals)))
    eps=1e-12
    complex_signal_power = np.mean(wideband_real_signals[idx]**2)
    complex_noise_power  = np.mean((wideband_real_signals[idx] - complex_rec_signal[idx])**2)
    complex_snr =  10 * np.log10((complex_signal_power + eps) / (complex_noise_power + eps))

    comp_freq_signal_power = np.mean(wideband_complex_signals[idx]**2)
    comp_freq_noise_power = np.mean((wideband_complex_signals[idx] - recovery_complex_signals[idx])**2)
    complex_freq_snr =  10 * np.log10((comp_freq_signal_power + eps) / (comp_freq_noise_power + eps))

    phase = np.random.uniform(0, 2*np.pi, len(recovery_mag_signals[idx]))
    X = recovery_mag_signals[idx] * np.exp(1j * phase)

    signal = np.real(ifft(ifftshift(X))) * len(wideband_freq)
    mag_signal_power = np.mean(wideband_mag_signals[idx]**2)
    mag_noise_power  = np.mean((wideband_mag_signals[idx] - recovery_mag_signals[idx])**2)
    mag_snr =  10 * np.log10((mag_signal_power + eps) / (mag_noise_power + eps))

    fig, axes = plt.subplots(2, 4, figsize=(8,4))  # 2 rows, 3 columns
    axes[0,0].plot(wideband_time, wideband_time_signals[idx])
    axes[0,0].set_title("Time (Ideal)")
    axes[0,1].plot(wideband_freq, wideband_mag_signals[idx])
    axes[0,1].set_title("Frequency - Magnitude (Ideal)")
    # axes[0,1].set_xlim(-300000, 300000)
    axes[1,1].plot(wideband_freq, wideband_phase_signals[idx])
    axes[1,1].set_title("Frequency - Phase (Ideal)")
    # axes[1,1].set_xlim(-300000, 300000)
    axes[1,2].plot(wideband_freq, wideband_real_signals[idx])
    axes[1,2].set_title("Frequency - Real (Ideal)")
    # axes[1,2].set_xlim(-300000, 300000)
    axes[0,2].plot(wideband_freq, wideband_imag_signals[idx])
    axes[0,2].set_title("Frequency - Imaginary (Ideal)")
    axes[1,3].plot(wideband_freq, recovery_real_signals[idx])
    axes[1,3].set_title("Frequency - Real (Ideal)")
    # axes[1,2].set_xlim(-300000, 300000)
    axes[0,3].plot(wideband_freq, recovery_imag_signals[idx])
    axes[0,3].set_title("Frequency - Imaginary (Ideal)")
    # axes[0,2].set_xlim(-300000, 300000)
    fig.suptitle("Simulated Analog Signals")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(8,4))  # 2 rows, 3 columns
    axes[0].plot(wideband_time, complex_rec_signal[idx])
    axes[0].set_title("Time (Ideal)")
    axes[1].plot(wideband_freq, recovery_real_signals[idx])
    axes[1].set_title("Frequency - Real (Ideal)")
    axes[2].plot(wideband_freq, recovery_imag_signals[idx])
    axes[2].set_title("Frequency - Imaginary (Ideal)")
    # axes[1].set_xlim(-300000, 300000)   
    # fig.suptitle("Simulated Analog Signals")
    fig.tight_layout()
    plt.show()

