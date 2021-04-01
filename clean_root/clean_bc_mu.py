from root_pandas import read_root
from root_pandas import to_root
import pandas as pd
import ROOT
import numpy as np

bc_file = "/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/BcToJPsiMuMu_is_jpsi_mu_merged.root"

df = read_root(bc_file,'BTo3Mu',where = 'mu1_isFromMuT & mu2_isFromMuT & mu1_isFromJpsi_MuT & mu2_isFromJpsi_MuT & k_isFromMuT', warn_missing_tree=True)
branches = [col for col in df.columns if 'hammer' in col]
for branch in branches:
    df[branch.replace('hammer_','')] = [ham if (not np.isnan(ham)) else 1. for ham in df[branch] ]
    
df.to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/BcToJPsiMuMu_is_jpsi_mu_trigger_hammer.root', key='BTo3Mu')
