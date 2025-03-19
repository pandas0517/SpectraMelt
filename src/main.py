'''
@author: pete
'''
if __name__ == '__main__':
    import json
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()
    with open(Path(os.getenv('SYSTEM_CONF')), 'r') as file:
        system_conf = json.load(file)
    with open(Path(os.getenv('FILEPATH_CONF')), 'r') as file:
        filepath_conf = json.load(file)
    pass