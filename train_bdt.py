import ROOT
from itertools import product
import matplotlib.pyplot as plt
import xgboost as xgb
# import modin.pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import pickle
from root_pandas import read_root, to_root

selection = ' & '.join([
    'mu1pt>3'               ,
    'mu2pt>3'               ,
    'kpt>2.5'               ,
    'abs(mu1eta)<2.5'       ,
    'abs(mu2eta)<2.5'       ,
    'abs(keta)<2.5'         ,
    'Bsvprob>1e-7'          ,
    'abs(k_dxy)<0.2'        ,
    'abs(mu1_dxy)<0.2'      ,
    'abs(mu2_dxy)<0.2'      ,
    'Bcos2D>0.95'           ,
    'Bmass<6.3'             ,
#     'mu1_mediumID>0.5'      ,
#     'mu2_mediumID>0.5'      ,
#     'k_mediumID>0.5'        ,
    'Bpt_reco>15'           ,
    'abs(mu1_dz-mu2_dz)<0.4', 
    'abs(mu1_dz-k_dz)<0.4'  ,
    'abs(mu2_dz-k_dz)<0.4'  ,
])

selection_mc = ' & '.join([
    selection                        ,
    'abs(k_genpdgId)==13'            ,
    '(abs(k_mother_pdgId)==541 | abs(k_mother_pdgId)==15)',
    'abs(mu1_genpdgId)==13'          ,
    'abs(mu1_mother_pdgId)==443'     ,
    'abs(mu2_genpdgId)==13'          ,
    'abs(mu2_mother_pdgId)==443'     ,
    'abs(mu1_grandmother_pdgId)==541',
    'abs(mu2_grandmother_pdgId)==541',
])

features = [
    'Bpt'         ,
    'Bmass'       ,
    'kpt'         ,
    'Bpt_reco'    ,
    'Blxy_sig'    ,
    'Bsvprob'     ,
#     'log10_svprob',
    'm_miss_sq'   ,
    'Q_sq'        ,
    'pt_var'      ,
    'pt_miss_vec' ,
    'E_mu_star'   ,
    'E_mu_canc'   ,
    'b_iso03_rel' ,
    'dr13'        ,
    'dr23'        ,
    'dr_jpsi_mu'  ,
]

samples = dict()
samples['tau'] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_jpsi_tau_merged.root').Filter(selection_mc) 
samples['mu' ] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_jpsi_mu_merged.root' ).Filter(selection_mc) 
samples['bkg'] = ROOT.RDataFrame('BTommm', '../dataframes_2020Oct19/Onia_merged.root'                          ).Filter(selection) 

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
    ('l1_iso04_rel', 'l1_iso04/mu1pt'       ),
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

print('adding new columns')
for k, v in samples.items():
    for new_column, new_definition in to_define:
        samples[k] = samples[k].Define(new_column, new_definition)

print('convert to pandas, exclude LorentzVector branches, as they do not cope well with pandas')
tau = pd.DataFrame(samples['tau'].AsNumpy(exclude=['mu1_p4', 'mu2_p4', 'mu3_p4', 'jpsi_p4']))
mu  = pd.DataFrame(samples['mu' ].AsNumpy(exclude=['mu1_p4', 'mu2_p4', 'mu3_p4', 'jpsi_p4']))
bkg = pd.DataFrame(samples['bkg'].AsNumpy(exclude=['mu1_p4', 'mu2_p4', 'mu3_p4', 'jpsi_p4']))

print('defining targets')
tau['target'] = np.ones ( tau.shape[0]).astype(np.int)
mu ['target'] = np.zeros( mu .shape[0]).astype(np.int)
bkg['target'] = np.full ((bkg.shape[0]), 2)

print('Splitting into train, validation and test...')
data = pd.concat([bkg, tau, mu])
# data = pd.concat([tau, mu])
data.sample(frac=1).reset_index(drop=True)
train, test = train_test_split(data, test_size=0.2, random_state=1986)
X_train, X_test = train[features], test[features]
y_train, y_test = train['target'], test['target']

print('Fitting BDT...')
    
clf = xgb.XGBClassifier(
#     n_jobs           = -1, # use all cores BROKWN in 1.2.0
    n_jobs           = 4, # use all cores
    max_depth        = 6,
    learning_rate    = 1e-1, # 0.01
    n_estimators     = 1000, # 400
    silent           = False,
    subsample        = 0.6,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    #gamma            = 0, # 20
    seed             = 1986,
    #scale_pos_weight = 1,
    #reg_alpha        = 0.01,
    #reg_lambda       = 1.2,
    objective        = 'multi:softprob',  # error evaluation for multiclass training
    num_class       = 3,
)

# http://rstudio-pubs-static.s3.amazonaws.com/368478_bf9700befeba4283a4640a9a1285af22.html
clf.fit(
    X_train, 
    y_train,
    eval_set              = [(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds = 20,
    eval_metric           = 'mlogloss',
#     eval_metric           = 'auc',
    verbose               = True,
#     nfold                 = 5, 
#     showsd                = TRUE, 
#     stratified            = TRUE, 
#     print.every.n = 10, early_stop_round = 20, maximize = FALSE, prediction = TRUE
    #sample_weight         = train['weight'],
)
 
flag = '24nov'   
pickle.dump(clf, open('bdtModel/BDT_Model_' +flag+ '.pck', 'wb'))
print('Model saved  ')
pickle.dump(features, open('bdtModel/BDT_Model_' +flag+ '_features.pck', 'wb'))
print('Features saved')


print('enrich the data')
for i, label in zip(range(3), ['mu', 'tau', 'bkg']):
    tau['bdt_%s' %label] = clf.predict_proba(tau[features])[:,i]
    mu ['bdt_%s' %label] = clf.predict_proba(mu [features])[:,i]
    bkg['bdt_%s' %label] = clf.predict_proba(bkg[features])[:,i]
    data['bdt_%s' %label] = clf.predict_proba(data[features])[:,i]

# tau['bdt'] = clf.predict_proba(tau[features])[:,1]
# mu ['bdt'] = clf.predict_proba(mu [features])[:,1]
# bkg['bdt'] = clf.predict_proba(bkg[features])[:,1]
# data['bdt'] = clf.predict_proba(data[features])[:,1]

mu .to_root('mu_bdt.root' , key='tree')
tau.to_root('tau_bdt.root', key='tree')
bkg.to_root('bkg_bdt.root', key='tree')

plt.clf()

xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
plt.plot(xy, xy, color='grey', linestyle='--')
# plt.xlim([10**-5, 1.0])
# plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# plt.xscale('log')

fpr, tpr, wps = roc_curve(data.target, data.bdt_bkg, pos_label=2)
plt.plot(fpr, tpr, label='bkg vs. all', color='b')

cuts_to_display = np.arange(0, 0.1, 0.02)

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps>icut).sum() - 1
    wp_x.append(fpr[idx])
    wp_y.append(tpr[idx])
    
plt.scatter(wp_x, wp_y, color='b')
for i, note in enumerate(cuts_to_display):
    plt.annotate('%.2f'%note, (wp_x[i], wp_y[i]))

fpr, tpr, wps = roc_curve(data.query('target!=0').target, data.query('target!=0').bdt_bkg, pos_label=2)
plt.plot(fpr, tpr, label='bkg vs. tau', color='b', linestyle='dashed')

fpr, tpr, wps = roc_curve(data.query('target!=1').target, data.query('target!=1').bdt_bkg, pos_label=2)
plt.plot(fpr, tpr, label='bkg vs. mu', color='b', linestyle='dotted')

fpr, tpr, wps = roc_curve(data.target, data.bdt_tau, pos_label=1)
plt.plot(fpr, tpr, label='tau vs. all', color='r')

fpr, tpr, wps = roc_curve(data.target, data.bdt_mu, pos_label=0)
plt.plot(fpr, tpr, label='mu vs. all', color='g')

fpr, tpr, wps = roc_curve(data.query('target!=2').target, data.query('target!=2').bdt_tau, pos_label=1)
plt.plot(fpr, tpr, label='tau vs. mu', color='orange', linestyle='dashed')

cuts_to_display = np.arange(0, 0.4, 0.05)

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps>icut).sum() - 1
    wp_x.append(fpr[idx])
    wp_y.append(tpr[idx])
    
plt.scatter(wp_x, wp_y, color='orange')
for i, note in enumerate(cuts_to_display):
    plt.annotate('%.2f'%note, (wp_x[i], wp_y[i]))


plt.legend()

plt.savefig('rocs.pdf')
