'''
@author: pete
'''
if __name__ == '__main__':
    from os import getenv
    from dotenv import load_dotenv
    from spectramelt.utils import get_logger, load_config_from_json
    from pathlib import Path
    from spectramelt.Recovery import Recovery
    from spectramelt.DataSet import DataSet

    load_dotenv()
    log_path = Path(getenv('SPECTRAMELT_LOG'))
    logger = get_logger(Path(__file__).stem, log_path)

    input_config = load_config_from_json(Path(getenv('INPUT_CONF')))
    nyfr_config = load_config_from_json(Path(getenv('NYFR_CONF')))
    recovery_config = load_config_from_json(Path(getenv('RECOVERY_CONF')))
    dataset = DataSet(input_config_name=input_config.get('config_name'),
                      DUT_config_name=nyfr_config.get('config_name'),
                      recovery_config_name=recovery_config.get('config_name'),
                      config_file_path=Path(getenv('DATASET_CONF')))
    dataset.set_config_name("Dataset_Config_2")
    dataset_dirs = dataset.get_directories()
    print(dataset.get_config_name())
    print(dataset_dirs['inputs'])