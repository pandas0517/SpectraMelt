'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import load_config_from_json, get_logger
    from pathlib import Path
    from spectramelt.NYFR import NYFR
    from spectramelt.DataSet import DataSet
    import atexit
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, ifft, fftshift, ifftshift

    load_dotenv()
    
    create_output_set = True

    create_nyfr_wave_params = True
    display_nyfr_signals = True
    
    create_premultiply_set = True
    display_premultiply_signals = True
    
    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))
    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      config_file_path=Path(getenv('DATASET_CONF')))

    directories = dataset.get_directories()
    input_dir = directories.get('inputs', "Inputs")
    output_dir = directories.get('outputs', "Outputs")    
    
    filenames = dataset.get_filenames()
    input_freq_signal_filename = filenames.get('input_freq_signal', "freq_signals.npy")
    output_signal_filename = filenames.get('output_signal', "time_signals.npy")