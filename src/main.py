'''
@author: pete
'''
if __name__ == '__main__':
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from NYFR import NYFR

    load_dotenv()

    with open(Path(os.getenv('FILEPATH_CONF')), 'r') as file:
        filepath_conf = json.load(file)
    nyfr = NYFR(Path(os.getenv('SYSTEM_CONF')))
    pass