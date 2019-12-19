import os
import sys

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

try:
    import cPickle as pickle
except:
    import pickle
from configuration import *


"""features = [
            "ww_pt","deltar_bbljj","deltaphi_bbljj",
            "bb_pt","deltaphi_ljj","deltar_ljj","mww","deltaphi_bb"
           ]"""
"""
features = [
            "deltaphi_bbljj",
            "deltaphi_ljj","deltaphi_bb"
           ]"""

#input_features_file = "./Data/signal_var_new.npy"

features = np.load(features_input_file)[0,:-1]
print("Features: {}\nNumber of features: {}".format(features,len(features)))


print("Features: {}\nNumber of features: {}".format(features,len(features)))

#input_file = "./Data/signal_new.npy"
"""
if(len(sys.argv)>1):
    input_file = sys.argv[1]
"""
print("Input_file: {}".format(data_input_file))

data_npy = np.load(data_input_file)[:,:-1]
data = pd.DataFrame(data_npy, columns = features)

X_train = data[features].values
print("X_train not standardize:\n{}".format(X_train))
print("X_train shape: {}".format(X_train.shape))

##Standardization

scaler = MinMaxScaler((-1,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)

print("X_train standardize:\n{}".format(X_train))

#Save the standardize data

output_filename = "./scaler/scaler_signal.pkl"

with open(output_filename,"wb") as output_scaler:
    pickle.dump(scaler, output_scaler)

print("Scaler saved as {}".format(output_filename))