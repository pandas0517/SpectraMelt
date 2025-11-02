'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger
    from pathlib import Path
    from spectramelt.InputSignal import InputSignal
    from spectramelt.DataSet import DataSet
    import atexit

    load_dotenv()

    logger = get_logger(Path(__file__).stem, Path(getenv('SPECTRAMELT_LOG')))

    input_signal = InputSignal(config_file_path=Path(getenv('INPUT_CONF')))

    dataset = DataSet(input_signal, config_file_path=Path(getenv('DATASET_CONF')))

    dataset.create_input_set()

    atexit.register(logger.info, "Completed Test\n")