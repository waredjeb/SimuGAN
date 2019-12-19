import os 
import ctypes
import numpy as np

#Create new directory
def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

#Smooth labels

def smooth_positive_labels(y):
    return y - 0.1 + (np.random.random(y.shape)*0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape)*0.1
