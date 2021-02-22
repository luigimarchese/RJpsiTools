'''
Prepare samples once and for all, adding BDT score and additional branches
'''

import ROOT
import numpy as np
import pickle
import pandas as pd
from root_pandas import to_root
from new_branches import to_define 

ROOT.EnableImplicitMT()

tree_dir = '/Users/manzoni/Documents/RJPsi/dataframes_december_2020'

features = pickle.load(open('bdtModel/BDT_Model_16Feb2021_10h32m50s_features.pck', 'rb'))
model    = pickle.load(open('bdtModel/BDT_Model_16Feb2021_10h32m50s.pck'         , 'rb'))

sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
#     'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
    'data'     ,
    'onia'     ,
]

samples = dict()
for isample_name in sample_names:
    samples[isample_name] = ROOT.RDataFrame('BTommm', '%s/BcToXToJpsi_is_%s_merged.root' %(tree_dir, isample_name))

# booleans are not recognised as numpy types and are not saved when the panda dataframe 
# is dumped to root
# ==> cast them to numpy.bool
to_cast = [
    'mu1_mediumID',
    'mu2_mediumID',
    'mu1_tightID',
    'mu2_tightID',
    'mu1_softID',
    'mu2_softID',
    'k_tightID',
    'k_mediumID',
    'k_softID',
    'mu1_isPF',
    'mu2_isPF',
    'k_isPF',
]

to_exclude = [
    'mu1_p4'     ,
    'mu2_p4'     ,
    'mu3_p4'     ,
    'kaon_p4'    ,
    'mmm_p4'     ,
    'jpsiK_p4'   ,
    'pion_p4'    ,
    'jpsipi_p4'  ,
    'jpsi_p4'    ,
    'Bdir_eta'   ,
    'Bdir_phi'   ,
]

for k, v in samples.items():
    
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)
    # convert to pandas
    samples[k] = pd.DataFrame(samples[k].AsNumpy(exclude=to_exclude))

    for icolumn in to_cast:
        samples[k][icolumn] = samples[k][icolumn].astype(np.bool, copy=False) 
    
    print('enrich the data', k)
    
    for i, label in zip(range(3), ['mu', 'tau', 'bkg']):
        samples[k]['bdt_%s' %label] = model.predict_proba(samples[k][features])[:,i]
    
    to_root(samples[k], '%s/BcToXToJpsi_is_%s_enriched.root' %(tree_dir, k), key='BTommm', store_index=False)



