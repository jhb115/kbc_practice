'''
This is script for running tests
'''


from pathlib import Path
import pkg_resources
import pickle
#from typing import Dict, Tuple, List



''

DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))

root = DATA_PATH / name

data = {}
for f in ['train', 'test', 'valid']:
    in_file = open(str(root / (f + '.pickle')), 'rb')
    data[f] = pickle.load(in_file)