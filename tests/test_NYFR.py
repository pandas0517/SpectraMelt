'''
@author: pete
'''
if __name__ == '__main__':
    import os
    import sys
    # Add the src directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from pathlib import Path
    from dotenv import load_dotenv
    from NYFR import NYFR
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    import numpy as np

    load_dotenv()

    nyfr = NYFR(file_path=Path(os.getenv('SYSTEM_CONF')))
    nyfr.initialize()
    input, _ = nyfr.create_input_signal(file_path=Path(os.getenv('WAVE_PARAMS')))
    output = nyfr.simulate_system(input_signal=input)
    dictionary = nyfr.create_dict()
    input_frequency = fft(input)
    output_frequency = fft(output)
    frequency = nyfr.get_frequncy_bins()
    sampled_frequency = nyfr.get_sampled_freq_bins()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(frequency, np.fft.fftshift(np.abs(input_frequency)))
    plt.xlim(-500,500)
    plt.subplot(2,1,2)
    plt.plot(sampled_frequency, np.fft.fftshift(np.abs(output_frequency)))
    plt.show()