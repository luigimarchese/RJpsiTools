import ROOT
from itertools import product
import matplotlib.pyplot as plt
import xgboost as xgb
import os
# import modin.pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import pickle
from root_pandas import read_root, to_root
from new_branches import to_define 
from selections import preselection, preselection_mc, pass_id, fail_id
from datetime import datetime

train_bdt = True
model_flag = '13Jul2021_11h10m53s'

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

features = [
    'Bpt'                  ,
    'Bmass'                ,
    'kpt'                  ,
    'Bpt_reco'             ,
    #'bvtx_log10_svprob'    ,
    #'jpsivtx_log10_svprob' ,
    #'bvtx_log10_lxy_sig'   ,
    #'jpsivtx_log10_lxy_sig',
    'mmm_p4_par'           ,
    'mmm_p4_perp'          ,
#     'm_miss_sq'            ,
#     'Q_sq'                 ,
    'pt_var'               ,
    'pt_miss_vec'          ,
    'pt_miss_scal'         ,
#     'E_mu_star'            ,
#     'E_mu_canc'            ,
#     'b_iso03_rel'          ,
    'b_iso04_rel'          ,
    'dr13'                 ,
    'dr23'                 ,
    'dr_jpsi_mu'           ,
    #'mcorr'                ,
    #'decay_time'           ,
    #'bct'                  ,
]

samples = dict()

tree_name = 'BTo3Mu'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'

samples['tau'] = ROOT.RDataFrame(tree_name, '%s/jpsi_tau_sf.root'     %tree_dir)
samples['mu' ] = ROOT.RDataFrame(tree_name, '%s/jpsi_mu_sf.root'      %tree_dir)
samples['cmb'] = ROOT.RDataFrame(tree_name, '%s/jpsi_x_mu_sf.root'    %tree_dir)    
samples['bkg'] = ROOT.RDataFrame(tree_name, '%s/data_fakerate.root'   %tree_dir)    

print('adding new columns')
for k, v in samples.items():
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)


tau = pd.DataFrame(samples['tau'].Filter(' & '.join([preselection_mc                  ])).AsNumpy())
mu  = pd.DataFrame(samples['mu' ].Filter(' & '.join([preselection_mc                  ])).AsNumpy())
cmb = pd.DataFrame(samples['cmb'].Filter(' & '.join([preselection_mc                  ])).AsNumpy()) # combinatorial bkg (already clean from signal)
bkg = pd.DataFrame(samples['bkg'].Filter(' & '.join([preselection   , fail_id])).AsNumpy())          # fakes

# merge together all backgrounds
bkg = pd.concat([bkg, cmb])
bkg.sample(frac=1).reset_index(drop=True)

print('defining targets')
tau['target'] = np.ones ( tau.shape[0]).astype(int)
mu ['target'] = np.zeros( mu .shape[0]).astype(int)
bkg['target'] = np.full ((bkg.shape[0]), 2)

# weights
# why 1000?
tau['w'] = np.ones(tau.shape[0]) * 1000./tau.shape[0]
mu ['w'] = np.ones(mu .shape[0]) * 1000./mu .shape[0]
bkg['w'] = np.ones(bkg.shape[0]) * 1000./bkg.shape[0]

#print(tau.shape[0],mu .shape[0],bkg.shape[0])

print('Splitting into train, validation and test...')
data = pd.concat([bkg, tau, mu])
# data = pd.concat([tau, mu])
data.sample(frac=1).reset_index(drop=True)
train, test = train_test_split(data, test_size=0.2, random_state=1986)
X_train, X_test = train[features], test[features]
y_train, y_test = train['target'], test['target']


clf = xgb.XGBClassifier(
    #     n_jobs           = -1, # use all cores BROKWN in 1.2.0
    n_jobs           = 4, # use all cores
    max_depth        = 6,
    learning_rate    = 1e-1, # 0.01
    n_estimators     = 400, # 400
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

flag = label   

if train_bdt:
    print('Fitting BDT...')
    
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
        sample_weight         = train['w'],
    )
    
    
    if not os.path.exists('bdtModel'):
        os.mkdir('bdtModel')
        
    pickle.dump(clf, open('bdtModel/BDT_Model_' +flag+ '.pck', 'wb'))
    print('Model saved  ')
    pickle.dump(features, open('bdtModel/BDT_Model_' +flag+ '_features.pck', 'wb'))
    print('Features saved')

# Load model and features
else:
    clf = pickle.load(open('bdtModel/BDT_Model_' +model_flag+ '.pck','rb'))
    features = pickle.load(open('bdtModel/BDT_Model_' +model_flag+ '_features.pck','rb'))

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
cuts_to_display = np.arange(0, 1, 0.1)

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

plt.savefig('rocs_%s.pdf' %flag)

