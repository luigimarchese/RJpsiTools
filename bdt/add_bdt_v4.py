'''
Adding BDT score and additional branches to the samples
'''
import math
import ROOT
import numpy as np
import pickle
import pandas as pd
from root_pandas import to_root, read_root
from new_branches import to_define 
from samples import sample_names_explicit_jpsimother_compressed as sample_names
#from sklearn.externals import joblib

ROOT.EnableImplicitMT()

flag = '09Mar2022_09h42m51s'

tree_name = 'BTo3Mu'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021'

classifier = pickle.load(open('bdt_models/%s/classifiers_%s.pck' %(flag,flag),'rb'))
features = pickle.load(open('bdt_models/%s/features_'%flag+flag+'.pck', 'rb'))

samples = dict()
for k in sample_names:
    samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_bdt_vv1.root' %(tree_dir, k) )
        

    #for k, v in samples.items():
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)
    # convert to pandas
    samples[k] = pd.DataFrame(samples[k].AsNumpy())

    #for icolumn in to_cast:
    #    if not math.isnan(samples[k][icolumn][0]):
    #        samples[k][icolumn] = samples[k][icolumn].astype(int)
    samples[k]['bdt_tau_mu_v2'] = np.zeros_like(samples[k].Bmass)

    bdt_proba = classifier.predict_proba(samples[k][features])[:,1]
    samples[k]['bdt_tau_mu_v2'     ] += bdt_proba
    print ('\t...done')
    print('enrich the data', k)

    to_root(samples[k], '%s/%s_bdt_vv1.root' %(tree_dir, k), key='BTo3Mu', store_index=False)



