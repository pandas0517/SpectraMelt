'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.NYFR import NYFR
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet

    load_dotenv()
    log_path = Path(getenv('SPECTRAMELT_LOG'))
    logger = get_logger(Path(__file__).stem, log_path)

    input_signal_1 = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))
    nyfr_1 = NYFR(config_file_path=Path(getenv('NYFR_CONF')))
    recovery_1 = Recovery(config_file_path=Path(getenv('RECOVERY_CONF')))
    dataset_1 = DataSet(nyfr_1,
                        input_signal_1,
                        recovery_1,
                        config_file_path=Path(getenv('DATASET_CONF')))
    dataset_dirs_1 = dataset_1.get_directories()
    print(dataset_1.get_config_name())
    print(dataset_dirs_1['inputs'])
    
    dataset_2 = DataSet()
    print(dataset_2.get_config_name())
    print(dataset_2.get_directories())