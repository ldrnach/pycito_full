import os
import numpy as np
import pycito.utilities as utils

SOURCE = os.path.join('data', 'a1_experiment','a1_hardware_data.pkl')

def get_data():
    return utils.load(SOURCE)

if __name__ == '__main__':
    print('Hello from common.py')