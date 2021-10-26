import os
import shutil

PATH_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))


def prepare_dictionaries():
    files = [f'{PATH_DATA}/american-english', f'{PATH_DATA}/british-english']
    for file in files:
        if not os.path.exists(f'/usr/share/dict/{file}'):
            shutil.copy(file, '/usr/share/dict/')
