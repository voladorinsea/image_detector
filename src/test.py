#-*- coding: UTF-8 -*-
from torch import tensor
import torch
import numpy as np
import argparse
import os
def parse_args():
    '''
    practice
    '''
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    
    help_info = "The path of address"
    parser.add_argument('--address', help=help_info)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    #args = parse_args()
    #print(args.address)
    #print(os.path.exists('imgs'))
    print(torch.__version__)