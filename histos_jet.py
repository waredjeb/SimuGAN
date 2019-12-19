import ROOT
import numpy as np
import math as mt
import argparse

from keras.models import load_model
import ctypes
import pickle
from help_fun import *
from configuration import *
import time

"""def mkdir_p(mypath):
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise"""



parser = argparse.ArgumentParser(
    description = 'train analysis')

parser.add_argument('-e','--epochs', default = 1000)
parser.add_argument('-b','--batch_size', default = 64)
parser.add_argument("-n",'--curr_epochs', default = 1000) #number of epochs to analyse
parser.add_argument("-r","--read_file_chi", default = 0)
#parser.add_argument('-d','--decay', default = 1) #1 (decay G), 11(decay both), 0(no decay)
args = parser.parse_args()

#n_epochs = int(args.epochs)
#batch_size = int(args.batch_size)
#decay = int(args.decay)
mod = ""


#Import true dataset
ROOT.gROOT.SetBatch(1)

X_true = np.load(data_input_file)
print("True dataset:\n",(X_true))

#Import scaler

with open(scaler_input,"rb") as scaler_file:
    scaler = pickle.load(scaler_file)
    

epochs = int(args.epochs)
batch_size = int(args.batch_size)
n_i = int(args.curr_epochs)
read_ni_min = int(args.read_file_chi)
#decay = int(args.decay)
#decay_v = 1e-5
#Import prediction dataset
"""epochs = 1000

batch_size = 32
decay = 11
decay_v = 1e-5"""
events = X_true.shape[0]
#generated_infile = "./gan_data/prediction/prediction_gan_ep-500000_trev-%i_bs%i_dec-%i-%f.npy"%(events,batch_size,decay,decay_v)
#X_gen = np.load(generated_infile)
#print("Generated dataset:\n",(X_gen))



    
#load model
mod_par = "bs_%i-ept_%i-trev_%i-%s"%(batch_size,epochs,events,extra)
main_dir_path = "results/"+mod_par
chi2_path = main_dir_path+"/chi2"

"""if(read_ni_min == 1):
    with open(chi2_path+"/chi2_min.txt","r") as f:
        args_min_chi = int(f.readline())
    f.close()
    n_i = int(args_min_chi * 5000)
print("n_i:{}".format(n_i))"""


num = int(n_i/5000)
print("NUM:{}".format(num))
model_infile = main_dir_path+"/models/%iep_%i.h5"%(num,n_i)
print("LOADING MODEL: {}".format(model_infile))
generator = load_model(model_infile)
generator.summary()
#X_noise = np.random.normal(0,1, size = [X_true.shape[0], 64])
start = time.time()
X_noise = np.random.normal(0,1, size = [X_true.shape[0], 64])
X_gen = generator.predict(X_noise)
X_gen = scaler.inverse_transform(X_gen)
end = time.time()


features = np.load(features_input_file)[0,:-1]
print("Features: {}\nNumber of features: {}".format(features,len(features)))

"""features = [
            "deltaphi_bb","ww_pt","deltar_bbljj","deltaphi_bbljj",
            "bb_pt","deltaphi_ljj","deltar_ljj","mww"
           ]"""

h = {}

nbins = 45

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

histos_path = main_dir_path+"/histos"
mkdir_p(histos_path)
name_root_outfile = histos_path+"/histos.root"
root_ofile = ROOT.TFile.Open(name_root_outfile,"RECREATE")


for i in range(0,len(features)):
    print("Loading feature: {}".format(features[i]))

    for j in range(0,X_true.shape[0]):

        h[features[i]].Fill(X_true[j,i])
        h_g[features[i]].Fill(X_gen[j,i])

    h[features[i]].Write()
    h_g[features[i]].Write()


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

c1 = ROOT.TCanvas("c1","c1",1000,1000)
c1.Divide(4,2)
leg = ROOT.TLegend()

leg.AddEntry(h[features[0]], "True")
leg.AddEntry(h_g[features[1]],"GAN")
for i in range(0,len(features)):
    c1.cd(i+1)
    h_g[features[i]].SetLineColor(ROOT.kRed)
    
    h_g[features[i]].Draw("h")
    h[features[i]].Draw("h same")    
    #chi2,ndf = PrintChi2(features[i])


print("Generation time: {}",.format(end - start))

leg.Draw("SAME")
c1.Draw()
c1.Write()
results_output = histos_path+"/histos_plot.pdf"
c1.SaveAs(results_output,"pdf")
root_ofile.Write()
root_ofile.Close()

