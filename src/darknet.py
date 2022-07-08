'''
implement the underlying architecture 
contains the code that creates the yolo network
'''
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    line = []
    with open(cfgfile, 'r') as conf:
        # split the file by '\n' character
        lines = conf.read().split('\n')
        lines = [x for x in lines if len(x)>0]
        lines = [x for x in lines if x[0]!='#']
        lines = [x.strip() for x in lines]
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key] = value
    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    # make sure the number of previous layer's filters
    prev_filter = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x["type"] == "convolutional":
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            padding = int("pad")
            activation = x["activation"]

            # pad规则
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add the Convolutional Layer
            conv = nn.Conv2d(prev_filter, filters, kernel_size, stride, pad)
            module.add_module("conv_{}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)
            
            # Add activation layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{}".format(index), activn)
            else: 
                pass

        if x["type"] == "upsample":
            stride = int("stride")
            








