'''
@author: pete
'''
if __name__ == '__main__':
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from NYFR import NYFR

    load_dotenv()

    nyfr = NYFR(file_path=Path(os.getenv('SYSTEM_CONF')))
    nyfr.initialize()
    output = nyfr.simulate_system(file_path=Path(os.getenv('WAVE_PARAMS')))
    dictionary = nyfr.create_dict()
    pass