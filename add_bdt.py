'''
Prepare samples once and for all, adding BDT score and additional branches
'''

import ROOT
import numpy as np
import pickle
import pandas as pd
from root_pandas import to_root

ROOT.EnableImplicitMT()

sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
    'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
]

samples = dict()
for isample_name in sample_names:
    samples[isample_name] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_%s_merged.root' %isample_name)

samples['data'] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/data_bc_mmm.root')

to_define = [
    ('abs_mu1_dxy' , 'abs(mu1_dxy)'         ),
    ('abs_mu2_dxy' , 'abs(mu2_dxy)'         ),
    ('abs_k_dxy'   , 'abs(k_dxy)'           ),
    ('abs_mu1_dz'  , 'abs(mu1_dz)'          ),
    ('abs_mu2_dz'  , 'abs(mu2_dz)'          ),
    ('abs_k_dz'    , 'abs(k_dz)'            ),
    ('log10_svprob', 'TMath::Log10(Bsvprob)'),
    ('b_iso03_rel' , 'b_iso03/Bpt'          ),
    ('b_iso04_rel' , 'b_iso04/Bpt'          ),
    ('k_iso03_rel' , 'k_iso03/kpt'          ),
    ('k_iso04_rel' , 'k_iso04/kpt'          ),
    ('l1_iso03_rel', 'l1_iso03/mu1pt'       ),
    ('l1_iso04_rel', 'l1_iso04/mu2pt'       ),
    ('l2_iso03_rel', 'l2_iso03/mu2pt'       ),
    ('l2_iso04_rel', 'l2_iso04/mu2pt'       ),
    ('mu1_p4'      , 'ROOT::Math::PtEtaPhiMVector(mu1pt, mu1eta, mu1phi, mu1mass)'),
    ('mu2_p4'      , 'ROOT::Math::PtEtaPhiMVector(mu2pt, mu2eta, mu2phi, mu2mass)'),
    ('mu3_p4'      , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, kmass)'),
    ('jpsi_p4'     , 'mu1_p4+mu2_p4'        ),
    ('jpsi_pt'     , 'jpsi_p4.pt()'         ),
    ('jpsi_eta'    , 'jpsi_p4.eta()'        ),
    ('jpsi_phi'    , 'jpsi_p4.phi()'        ),
    ('jpsi_mass'   , 'jpsi_p4.mass()'       ),
    ('dr12'        , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu2_p4.Vect())'),
    ('dr13'        , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu3_p4.Vect())'),
    ('dr23'        , 'ROOT::Math::VectorUtil::DeltaR(mu2_p4.Vect(), mu3_p4.Vect())'),
    ('dr_jpsi_mu'  , 'ROOT::Math::VectorUtil::DeltaR(jpsi_p4.Vect(), mu3_p4.Vect())'),
    # is there a better way?
    ('maxdr'       , 'dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'),
    ('mindr'       , 'dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'),
]

features = pickle.load(open('bdtModel/BDT_Model_24nov_features.pck', 'rb'))
model = pickle.load(open('bdtModel/BDT_Model_24nov.pck', 'rb'))

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

for k, v in samples.items():
    
    for new_column, new_definition in to_define:
        samples[k] = samples[k].Define(new_column, new_definition)
    # convert to pandas
    samples[k] = pd.DataFrame(samples[k].AsNumpy(exclude=['mu1_p4', 'mu2_p4', 'mu3_p4', 'jpsi_p4']))

    for icolumn in to_cast:
        samples[k][icolumn] = samples[k][icolumn].astype(np.bool, copy=False) 
    
    print('enrich the data', k)
    
    for i, label in zip(range(3), ['mu', 'tau', 'bkg']):
        samples[k]['bdt_%s' %label] = model.predict_proba(samples[k][features])[:,i]
    
    to_root(samples[k], '../samples_20_novembre/samples/BcToXToJpsi_is_%s_enriched.root' %k, key='BTommm', store_index=False)



