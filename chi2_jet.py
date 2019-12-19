import ROOT
import numpy as np
import math as mt
import argparse

from keras.models import load_model
import ctypes
import pickle
import glob
from tqdm import tqdm

from help_fun import *
from configuration import *

def PrintChi2(hname):
    chi2 =ROOT.Double(0.)
    ndf = ctypes.c_int(0)
    igood = ctypes.c_int(3)

    #chi2 = _h_mc[hname].Chi2Test(_h[hname], "WW CHI2/NDF")
    h[hname].Chi2TestX(h_g[hname], chi2, ndf, igood, "WW")
    ndf = ndf.value
    l = ROOT.TLatex()
    l.SetNDC()
    l.SetTextFont(42)
    l.SetTextSize(0.04)
    txt = "#chi^{2}/NDF = %.1f/%i = %.1f" % (chi2, ndf, chi2/ndf)
    l.DrawLatex(0.3, 0.87, txt)

    return chi2, ndf


parser = argparse.ArgumentParser(
    description = 'train analysis')

parser.add_argument('-e','--epochs', default = 1000)
parser.add_argument('-b','--batch_size', default = 64)
#parser.add_argument('-d','--decay', default = 1) #1 (decay G), 11(decay both), 0(no decay)
args = parser.parse_args()

epochs = int(args.epochs)
batch_size = int(args.batch_size)


X_true = np.load(data_input_file)
print("True dataset:\n",(X_true))


#input_features_file = "./Data/signal_var_new.npy"

features = np.load(features_input_file)[0,:-1]
print("Features: {}\nNumber of features: {}".format(features,len(features)))


events = 60000


mod_par = "bs_%i-ept_%i-trev_%i-%s"%(batch_size,epochs,X_true.shape[0],extra)
main_dir_path = "results/"+mod_par
models_path = main_dir_path+"/models"
models = glob.glob(models_path+"/*.h5")
#print(models_path)
#models = glob.glob("gan_data/models/model_trained_gan_bs64_dec-0-0.000010-Adam-m2/*.h5")
models = sorted(models, reverse = False)
print(models_path)
tot_mod = len(models)
print(models)

x_axis = np.arange(5000,(len(models)+1)*5000,5000)
print(x_axis)


rdn_ind = np.random.randint(0,X_true.shape[0],size = events)
X_true = X_true[rdn_ind,:]

#Import true dataset
ROOT.gROOT.SetBatch(1)


#Import scaler

with open(scaler_input,"rb") as scaler_file:
    scaler = pickle.load(scaler_file)
    
#Import prediction dataset

events = 10000
CHI2 = []

histos_path = main_dir_path+"/histos"

step = 5
pbar = tqdm(range(0,int((tot_mod+1)/step)))
for i in range(1,len(models)+1,step):

    name = models_path+"/%iep_%i.h5"%(i,int(i*5000))
    print(name)
    chi2_tot = 0
    generator = load_model(name)
    generator.summary()
    X_noise = np.random.normal(0,1, size = [X_true.shape[0], 64])
    X_gen = generator.predict(X_noise)
    X_gen = scaler.inverse_transform(X_gen)

    h = {}

    nbins = 100

    h["j1_m"] = ROOT.TH1F(
        "j1_m","Jet_1 Mass;Mass [GeV/c^2]; Events",nbins,-0.02,0.02)

    h["j2_m"] = ROOT.TH1F(
        "j2_m","Jet_2 Mass;Mass [GeV/c^2]; Events",nbins,-0.02,0.02)

    h["j1_pt"] = ROOT.TH1F(
        "j1_pt", "Jet_1 transverse momentum;Transverse momentum; Events",nbins,0,500
    )


    h["j2_pt"] = ROOT.TH1F(
        "j2_pt", "Jet_2 transverse momentum;Transverse momentum; Events",nbins,0,500
    )

    h["j1_eta"] = ROOT.TH1F(
        "j1_eta", "Jet_1 pseudorapidity;#eta; Events",nbins,-3,3
    )

    h["j2_eta"] = ROOT.TH1F(
        "j2_eta", "Jet_2 pseudorapidity;#eta; Events",nbins,-3,3
    )


    h_g = {}

    h_g["j1_m"] = ROOT.TH1F(
        "j1_m","Jet_1 Mass;Mass [GeV/c^2]; Events",nbins,-0.02,0.02)

    h_g["j2_m"] = ROOT.TH1F(
        "j2_m","Jet_2 Mass;Mass [GeV/c^2]; Events",nbins,-0.02,0.02)

    h_g["j1_pt"] = ROOT.TH1F(
        "j1_pt", "Jet_1 transverse momentum;Transverse momentum; Events",nbins,0,500
    )


    h_g["j2_pt"] = ROOT.TH1F(
        "j2_pt", "Jet_2 transverse momentum;Transverse momentum; Events",nbins,0,500
    )

    h_g["j1_eta"] = ROOT.TH1F(
        "j1_eta", "Jet_1 pseudorapidity;#eta; Events",nbins,-3,3
    )

    h_g["j2_eta"] = ROOT.TH1F(
        "j2_eta", "Jet_2 pseudorapidity;#eta; Events",nbins,-3,3
    )

    name_root_histo_file = histos_path+"/histos_%i.root"%(int(i*5000))
    
    root_ofile = ROOT.TFile.Open(name_root_histo_file,"RECREATE")
    for i in range(0,len(features)):
        print("Loading feature: {}".format(features[i]))

        for j in range(0,X_true.shape[0]):

            h[features[i]].Fill(X_true[j,i])
            h_g[features[i]].Fill(X_gen[j,i])

        h[features[i]].Write()
        h_g[features[i]].Write()


    for i in range(0,len(features)):            
        #h_g[features[i]].SetLineColor(ROOT.kRed)
        #h[features[i]].Draw("h")    
        #h_g[features[i]].Draw("h same")
        chi2,ndf = PrintChi2(features[i])
        print("NDF:{}".format(ndf))
        chi2_tot += chi2/ndf
    CHI2.append(chi2_tot)

    pbar.update()
    root_ofile.Write()
    root_ofile.Close()
pbar.close()

import matplotlib.pyplot as plt
chi2_path = main_dir_path+"/chi2"
mkdir_p(chi2_path)
fig = plt.figure(figsize = (10,8))
#plt.plot(x_axis,CHI2,'.')
plt.plot(CHI2,'.')
plt.xlabel("epochs")
plt.ylabel("$\chi^2$")
plt.savefig(chi2_path+"/chi2.pdf")
#plt.show()

Chi2_np = np.array(CHI2)
chi2_min_ind = np.argmin(Chi2_np)

with open(chi2_path+'/chi2_min.txt', 'w') as f:
    f.write('%i' % chi2_min_ind)
