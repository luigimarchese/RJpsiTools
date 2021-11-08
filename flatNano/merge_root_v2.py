#Script that merges the flat root files from the same collection
import os
import subprocess
import pandas as pd
from personal_settings import *
import ROOT
import math
from root_pandas import to_root, read_root

path_or = personal_tier_path+'dataframes_2021Oct22/'

#dataset = 'HbToJPsiMuMu_3MuFilter/'
dataset = 'BcToJPsiMuMu/'

if not ("BcToJPsiMuMu") in dataset:
    flag_names = ['ptmax']

else:
    flag_names = ['is_jpsi_tau','is_jpsi_mu','is_jpsi_pi','is_psi2s_mu','is_chic0_mu','is_chic1_mu','is_chic2_mu','is_hc_mu','is_psi2s_tau','is_jpsi_3pi','is_jpsi_hc']
    
for flag in flag_names:
    path = path_or

    #Print the file names
    if ("BcToJPsiMuMu") in dataset:
        lsOut = subprocess.getoutput('ls ' + path + dataset + flag)
    else:
        lsOut = subprocess.getoutput('ls ' + path + dataset )
    #print(lsOut)
    files = lsOut.split('\n')
    
    print("%s files are going to be merged" %(len(files)))
    
    #we need to check that the first file that we put as Source is ok, otherwise the fnal merged fie will nto be ok (I know frome xperience, i.e. psi2s_tau)

    for file in files:
        if '3m' in file:
            f = read_root(path + dataset + flag + '/' +file,"BTo3Mu")
            if not math.isnan(f.jpsi_mass[0]):
                first_file = file
                break
    os.system('mv '+ path + dataset + flag + '/' +first_file+' '+path + dataset+'.')
    os.system('hadd '+ path + dataset.strip('/') + '_'+flag + '_merged_v5.root '+path + dataset+ first_file +' '+path + dataset + flag + '/*')        
    os.system('mv '+ path + dataset +  '/' +first_file+' '+path + dataset+flag +'/.')
