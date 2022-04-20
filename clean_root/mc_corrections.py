from root_pandas import read_root
from root_pandas import to_root
from samples import sample_names
#from samples import sample_names_explicit_jpsimother_compressed as sample_names
#from samples import jpsi_x_mu_sample_jpsimother_splitting_compressed as jpsi_x_mu_samples
import pandas as pd
import ROOT
import sys
import numpy as np

# Path for final root files 
path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021'

input_map_path = '/work/lmarches/CMS/RJPsi_Tools/CMSSW_10_6_14/src/RJpsiTools/plotting/ReweightingOutput/Maps.root'

input_file = ROOT.TFile.Open(input_map_path,'r') 
histo = input_file.Get("Histo_weightsPt")
histo_eta = input_file.Get("Histo_weightsEta")
histo_pteta = input_file.Get("Histo_weightsPtEta")

for sname in sample_names[:-1]:

    df = read_root(path+'/'+sname+'_with_mc_corrections.root','BTo3Mu',warn_missing_tree=True)    
    df.index= [i for i in range(len(df))]

    print(df['bc_gen_pt'])
    weights_pteta = []
    weights_eta = []
    weights_pt = []
    for i in range(len(df)):
        #pt eta 2D
        bx = histo_pteta.GetXaxis().FindBin(df.bc_gen_pt[i])
        by = histo_pteta.GetYaxis().FindBin(df.bc_gen_eta[i])
        w = histo_pteta.GetBinContent(bx,by)
        # pt and eta 1D
        beta = histo_eta.GetXaxis().FindBin(df.bc_gen_eta[i])
        bpt = histo.GetXaxis().FindBin(df.bc_gen_pt[i])
        weta = histo_eta.GetBinContent(beta)
        wpt = histo.GetBinContent(bpt)

        #print(w)
        if weta == 0:
            weta = 1.
        if wpt == 0:
            wpt = 1.
        if w ==0:
            w =1.
        weights_eta.append(weta)
        weights_pt.append(wpt)
        weights_pteta.append(w)
        #print(weight)
    
    #print(sum(weights) / len(weights))
    '''df['mc_correction_pteta_weight'] = weights
    df['mc_correction_weight_pteta_norm'] = df['mc_correction_pteta_weight']/norm
    '''
    df['mc_correction_gen_eta_weight'] = weights_eta
    df['mc_correction_gen_pt_weight'] = weights_pt
    df['mc_correction_gen_pteta_weight'] = weights_pteta

    df.to_root(path+'/'+sname+'_with_mc_corrections.root', key='BTo3Mu')
