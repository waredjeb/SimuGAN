import os
import sys
import csv
import argparse
import random, math
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.optimizers import *
from keras import regularizers
from keras.callbacks import *
from keras.utils import plot_model

import pandas as pd

from models import *
from help_fun import *
from configuration import *


parser = argparse.ArgumentParser(
    description = 'train analysis')

parser.add_argument('-e','--epochs', default = 1000)
parser.add_argument('-b','--batch_size', default = 64)
#parser.add_argument('-d','--decay', default = 1) #1 (decay G), 11(decay both), 0(no decay)
args = parser.parse_args()

n_epochs = int(args.epochs)
batch_size = int(args.batch_size)
#decay = int(args.decay)

print("\n\n############## RUNNING TRAIN: Epochs-{}, batch_size-{}################### \n\n".format(n_epochs,batch_size))

#input_file = "./Data/signal_new.npy"
#input_features_file = "./Data/signal_var_new.npy"

features = np.load(features_input_file)[0,:-1]
print("Features: {}\nNumber of features: {}".format(features,len(features)))

print("Input_file: {}".format(data_input_file))

#data_npy = np.load(input_file)[:,0]
data_npy = np.load(data_input_file)[:,:-1]
data = pd.DataFrame(data_npy, columns = features)

X_train = data[features].values
print("X_train not standardize:\n{}".format(X_train))


#scaler_name = "./scaler/scaler_signal.pkl"
print("Loading scaler: {}".format(scaler_input))

with open(scaler_input,'rb') as scaler_in:
    scaler = pickle.load(scaler_in)

X_train = scaler.transform(X_train)

#Starting with the models and the train

#Optimizer Generator
#wanted_variables = np.array(["j1_m","j2_m","j1_pt","j2_pt","j1_eta","j2_eta"])extra = "Adam-m3-b1_05-b2_08-gaus"
lr = 0.1

Gadam_opt = Adam(lr = 0.001, beta_1 = 0.5, beta_2 = 0.8)
    #Optimizer Discriminator
Dadam_opt = SGD(lr = lr)

noise_input_size = 64
Generator = generator_conv(input_size = noise_input_size, output_size = len(features))
print("Generator model built!\n ####GENERATOR####")
Generator.name = "Generator"
Generator.compile(loss = 'mean_squared_error', optimizer = Gadam_opt)
Generator.summary()

Discriminator = discriminator_conv(output_size = len(features))
print("Discriminator model built!\n ####DISCRIMINATOR####")
Discriminator.name = 'Discriminator'
Discriminator.compile(loss = 'binary_crossentropy', optimizer = Dadam_opt,
                            metrics = ['accuracy'])
Discriminator.summary()

#To train only the generator
Discriminator.trainable = False
Ogen_in = Input(shape = (noise_input_size,))
Ogen = Generator(Ogen_in)
Ogen_out = Discriminator(Ogen)
Ogenerator = Model(Ogen_in,Ogen_out)
Ogenerator.name = "Only-generator"
Ogenerator.compile(
    loss = 'binary_crossentropy', optimizer = Gadam_opt
)
Ogenerator.summary()

train_events = X_train.shape[0]
mod_par = "bs_%i-ept_%i-trev_%i-%s"%(batch_size,n_epochs,train_events,extra)
main_dir_path = "results/"+mod_par
mkdir_p(main_dir_path) #Main directory for the current training.

model_plot_path = main_dir_path+"/plot_model"
mkdir_p(model_plot_path)

plot_model(Generator , model_plot_path+"/generator.png")
plot_model(Discriminator,model_plot_path+"/discriminator.png")
plot_model(Ogenerator, model_plot_path+'/Gan.png')


models_path = main_dir_path+"/models"
mkdir_p(models_path)
#Extracting 'train_events' random from X_train 
#Random indices
train_index = random.sample(range(0,X_train.shape[0]),train_events)
#Random events
X_train_true = X_train[train_index,:]


#Creating the noise sample for the generator input
#Half true and half fake!
X_noise = np.random.normal(0,1, size = [X_train_true.shape[0],noise_input_size])
X_train_false = Generator.predict(X_noise) #Generator not trained yet, the prediction are almost random

#Final training dataset!
X = np.concatenate((X_train_true,X_train_false))
#Need the labels for the discriminator!
N = X_train_false.shape[0]
y = np.zeros([2*N])
#y_t = np.random.uniform(0.8,1.2, size = [N,])
#y_f = np.random.uniform(0,0.3, size = [N,])
#y = np.concatenate((y_t,y_f))
#print("Y:{}".format(y))
y[:N] = 1 #TRUE!
y[N:] = 0 #FAKE!

#Train the discriminator! Only a pre-train! Few epochs!
Discriminator.trainable = True
#Discriminator.fit(X,y, epochs = 1, batch_size= 64)

#Need several losses and accuracies!

history = {
        "d_loss": [], "g_loss": [],
        "d_acc": [],  "g_acc": [],
        "d_loss_t":[], "d_loss_f":[],
        "g_loss_t":[], "g_loss_f":[],
        "d_acc_t": [], "d_acc_f":[],
        "d_lr": [], "g_lr":[]
}


from tqdm import tqdm
#Starting with the train!
num = 0
y_true = np.ones((batch_size,1))
y_false = np.zeros((batch_size,1))

#y_true = np.random.uniform(0.8,1.2, size = [batch_size,1])
#y_false = np.random.uniform(0,0.3, size = [batch_size,1])

pbar = tqdm(range(0,n_epochs+1))
for n_i in range(n_epochs+1):

    d_lr = K.get_value( Discriminator.optimizer.lr)
    #print("Discriminator d_lr:",d_lr)
    g_lr = K.get_value(Generator.optimizer.lr)

    #Some real events
    train_index = random.sample(range(0,X_train.shape[0]),batch_size)
    X_train_true = X_train[train_index,:]

    X_noise = np.random.normal(0,1,size=[batch_size,noise_input_size])
    X_train_false = Generator.predict(X_noise)

    Discriminator.trainable = True

    d_loss_t, d_loss_f = Discriminator.train_on_batch(X_train_true,y_true)
    d_acc_t,d_acc_f = Discriminator.train_on_batch(X_train_false,y_false)

#The total loss function is computed as the mean between the real and the fake

    d_loss = 0.5*np.add(d_loss_t,d_loss_f)
    d_acc = 0.5*np.add(d_acc_t,d_acc_f)

    history["d_loss"].append(d_loss)
    history["d_acc"].append(d_acc)
    history["d_loss_t"].append(d_loss_t)
    history["d_loss_f"].append(d_loss_f)
    history["d_acc_t"].append(d_acc_t)
    history["d_acc_f"].append(d_acc_f)
    history["d_lr"].append(d_lr)
    history["g_lr"].append(g_lr)

#Train generator

    Discriminator.trainable = False

    g_loss = Ogenerator.train_on_batch(X_noise,y_true)

    history["g_loss"].append(g_loss)

    if(n_i % 5000 == 0 and n_i != 0): #Save model every 5k epochs
        num += 1
        print("Discriminator d_lr:",d_lr)
        print("Generator d_lr:",K.get_value(Generator.optimizer.lr))
        print("Loss discriminator: {}".format(d_loss))
        print("Loss generator: {}".format(g_loss))
        print("Loss discriminator true {}".format(d_loss_t))
        print("Loss discriminator false {}".format(d_loss_f))
        model_output_generator = models_path+"/%iep_%i.h5"%(num,n_i)
        #model_output_generator = "gan_data/models/model_trained_gan_bs64_dec-0-0.000010-Adam-m2/%imodel_trained_gan_ep-%i_trev-%i_bs%i_dec-%i-%f-%s.h5"%(num,n_i,train_events,batch_size,decay,decay_v,extra)
        print("Saved Model:",(model_output_generator))
        Generator.save(model_output_generator)
        
    pbar.update()
pbar.close()


data = pd.DataFrame(data_npy, columns = features)
X_train = data[features].values
print(X_train)

X_noise = np.random.normal(0,1, size = [X_train.shape[0],noise_input_size])

prediction = Generator.predict(X_noise)

prediction = scaler.inverse_transform(prediction)
print("Prediction:\n{}".format(prediction))
print("Prediction shape:{}".format(prediction.shape))

model_output_generator = models_path+"/%iep_%i.h5"%(num,n_i)
#model_output_generator = "gan_data/models/model_trained_gan_ep-%i_trev-%i_bs%i_dec-%i-%f-%s.h5"%(n_epochs,train_events,batch_size,decay,decay_v,extra)
Generator.save(model_output_generator)

train_path = main_dir_path+"/training"
mkdir_p(train_path)
output_train = train_path+"/training.pickle"


with open(output_train, 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


prediction_path = main_dir_path+"/prediction"
mkdir_p(prediction_path)
np.save(prediction_path+"/prediction.npy",prediction)


