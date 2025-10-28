'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from dotenv import load_dotenv
    load_dotenv()  
    # Signal parameters
    fs_in = 44100  # Input sampling frequency (e.g., 44.1 kHz)
    fs_sh = 500    # Sample and hold frequency (e.g., 500 Hz)
    duration = 1.0 # seconds
    t_in = np.linspace(0, duration, int(fs_in * duration), endpoint=False)

    # Generate an input signal (sum of sinusoids)
    f1, f2 = 50, 150
    signal_in = 0.7 * np.sin(2 * np.pi * f1 * t_in) + 0.3 * np.sin(2 * np.pi * f2 * t_in) + 2.5 # Sine wave between 0V to 5V

    # ADC parameters
    bits = 8
    v_ref_min_adc = 0.0
    v_ref_max_adc = 5.0
    thermal_noise_strength = 0.05 # Standard deviation of thermal noise
    
    # Simulate the realistic sample and hold
    output_signal, sh_indices, sampled_values = realistic_sample_and_hold(
        signal_in,
        fs_in,
        fs_sh,
        non_linearity_mode="tanh",
        alpha=0.5,
        jitter_std=10e-6,
        acquisition_time_constant=1e-5,
        hold_noise_std=0.01,
        v_min=v_ref_min_adc,
        v_max=v_ref_max_adc
    )
    
    sampled_time = t_in[sh_indices]

    # --- Plotting the S&H results ---
    plt.figure(figsize=(15, 8))
    plt.plot(t_in, signal_in, label='Original Analog Signal', color='gray', linestyle='--')
    plt.plot(t_in, output_signal, label='Realistic S&H Output', color='red', alpha=0.7)
    plt.plot(t_in[sh_indices], signal_in[sh_indices], 'x', markersize=8, color='blue', label='Jittered Sample Points')
    plt.title('Realistic Sample and Hold with Non-linear Distortion and Imperfections')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.ylim(1.5, 3.5)
    plt.xlim(0.1, 0.15) # Zoom in to see the effects clearly
    plt.show()

    # Quantize the signal with thermal noise
    quantized_signal_with_noise, mid_times, adc_codes_with_noise = realistic_adc_quantizer(
        sh_output_signal=output_signal, 
        sh_indices=sh_indices,
        fs_in=fs_in,
        num_bits=bits,
        v_ref_min=v_ref_min_adc,
        v_ref_max=v_ref_max_adc,
        thermal_noise_std_dev=thermal_noise_strength
    )

    # Quantize the signal without noise for comparison
    quantized_signal_ideal, _, adc_codes_ideal = realistic_adc_quantizer(
        sh_output_signal=output_signal, 
        sh_indices=sh_indices,
        fs_in=fs_in,
        num_bits=bits,
        v_ref_min=v_ref_min_adc,
        v_ref_max=v_ref_max_adc,
        thermal_noise_std_dev=0
    )

    print(f"Original signal sample: {signal_in[100]:.4f}V")
    print("First 10 midpoint times:", mid_times[:10])
    print("First 10 quantized voltages with noise:", quantized_signal_with_noise[:10])
    print("First 10 ADC n-bit codes with noise:", adc_codes_with_noise[:10])
    print("First 10 quantized voltages:", quantized_signal_ideal[:10])
    print("First 10 ADC n-bit codes:", adc_codes_ideal[:10])
    
    # --- Plotting the Quantization results ---
    # --- Side-by-side subplots for comparison ---
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(t_in, output_signal, color='black')
    axs[0].set_title('Original S&H Output Signal')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].grid(True)
    axs[0].set_ylim(1.5, 3.5)
    axs[0].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    axs[1].step(mid_times, quantized_signal_ideal, color='green', where='mid')
    axs[1].set_title('Quantized Signal (Ideal)')
    axs[1].set_ylabel('Voltage (V)')
    axs[1].grid(True)
    axs[1].set_ylim(1.5, 3.5)
    axs[1].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    axs[2].step(mid_times, quantized_signal_with_noise, color='orange', where='mid')
    axs[2].set_title('Quantized Signal (With Thermal Noise)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Voltage (V)')
    axs[2].grid(True)
    axs[2].set_ylim(1.5, 3.5)
    axs[2].set_xlim(0.1, 0.15) # Zoom in to see the effects clearly
    
    plt.tight_layout()
    plt.ylim(1.5, 3.5)
    plt.xlim(0.1, 0.15) # Zoom in to see the effects clearly
    plt.show()