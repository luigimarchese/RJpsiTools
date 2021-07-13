'''
Prepare samples once and for all, adding BDT score and additional branches
'''
import math
import ROOT
import numpy as np
import pickle
import pandas as pd
from root_pandas import to_root, read_root
from new_branches import to_define 
from samples import sample_names

ROOT.EnableImplicitMT()

flag = '13Jul2021_15h39m42s'

tree_name = 'BTo3Mu'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'

features = pickle.load(open('bdtModel/BDT_Model_'+flag+'_features.pck', 'rb'))
model    = pickle.load(open('bdtModel/BDT_Model_'+flag+'.pck'         , 'rb'))

samples = dict()
for isample_name in sample_names:
    if isample_name != 'data':
        samples[isample_name] = ROOT.RDataFrame(tree_name,'%s/%s_sf.root' %(tree_dir, isample_name) )
    else:
        samples[isample_name] = ROOT.RDataFrame(tree_name, '%s/%s_fakerate.root' %(tree_dir, isample_name))
        
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
    #'mu1_isPF',
    #'mu2_isPF',
    #'k_isPF',
]

for k, v in samples.items():
#for k in ['psi2s_tau']:
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)
    # convert to pandas
    samples[k] = pd.DataFrame(samples[k].AsNumpy())

    for icolumn in to_cast:
        if not math.isnan(samples[k][icolumn][0]):
            samples[k][icolumn] = samples[k][icolumn].astype(int)
 
    print('enrich the data', k)
    for i, label in zip(range(3), ['mu', 'tau', 'bkg']):
        samples[k]['bdt_%s' %label] = model.predict_proba(samples[k][features])[:,i]
    
    to_root(samples[k], '%s/%s_bdtenriched.root' %(tree_dir, k), key='BTo3Mu', store_index=False)



