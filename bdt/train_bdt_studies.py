'''
BDT and Grid Search
'''

import ROOT
from itertools import product
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
import xgboost as xgb
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from xgboost import plot_tree

import pandas as pd
import numpy as np
import pickle

from root_pandas import read_root, to_root
from new_branches import to_define 
from selections import preselection, preselection_mc, pass_id, fail_id
from datetime import datetime

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)

# No need to retrain the BDT if it is already done
train_bdt = True
model_flag = '13Jul2021_11h10m53s' #used only if train_bdt = False

grid_search = False

flag = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

features = [
    'mmm_p4_par'           ,
    'mmm_p4_perp'          ,
    'pt_var'               ,
    'pt_miss_vec'          ,
    'Q_sq'                 ,
    'E_mu_star'            ,
    'm_miss_sq'            ,
    'mcorr'                ,
    'decay_time_ps'        ,

    #new
    #'Bmass',
    #'mu1_dxy_sig',
    #'mu2_dxy_sig',
    #'k_dxy_sig',
    #'mu1_dz_sig',
    #'mu2_dz_sig',
    #'k_dz_sig',
    #'bvtx_lxy_sig',
    #'bvtx_cos2D',
    #'dr12',
    #'dr13',
    #'dr23',
    #'ip3d_sig' 
]

samples = dict()

tree_name = 'BTo3Mu'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'

#samples['tau'] = ROOT.RDataFrame(tree_name, '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Sep22/BcToJpsiTauNu_ptmax_merged.root')
samples['tau'] = ROOT.RDataFrame(tree_name, '%s/jpsi_tau_sf.root'     %tree_dir)
samples['bkg'] = ROOT.RDataFrame(tree_name, '%s/jpsi_x_mu_sf.root'    %tree_dir)    #combinatorial bkg

print('adding new columns')
for k, v in samples.items():
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)

tau = pd.DataFrame(samples['tau'].Filter(' & '.join([preselection_mc                  ])).AsNumpy())
bkg = pd.DataFrame(samples['bkg'].Filter(' & '.join([preselection_mc                  ])).AsNumpy()) 

print("Number of events : ", tau.shape[0], bkg.shape[0])

print('defining targets')
tau['target'] = np.ones ( tau.shape[0]).astype(int)
bkg['target'] = np.zeros( bkg.shape[0]).astype(int)

minn = min(tau.shape[0],bkg.shape[0])
tau['w'] = np.ones(tau.shape[0]) * minn/tau.shape[0]
bkg['w'] = np.ones(bkg.shape[0]) * minn/bkg.shape[0]

data = pd.concat([bkg, tau])
data = data.reset_index()

train, test = train_test_split(data, test_size=0.25, random_state=1986)
X_train, X_test = train[features], test[features]
y_train, y_test = train['target'], test['target']

'''clf = xgb.XGBClassifier(
    use_label_encoder =False, #needed for a Warning
    eval_metric = 'auc', #needed for a Warning
    learning_rate= 0.05, 
    max_depth= 7, 
    n_estimators= 50,
    colsample_bytree= 0.6, 

)
'''

clf = xgb.XGBClassifier(
    use_label_encoder =False,
    eval_metric = 'error',
    learning_rate    = 2e-2, 
    n_estimators     = 15000,
    colsample_bytree = 0.5,
    max_depth = 6,
    reg_alpha = 0.3,
    reg_lambda = 1,
    #eval_metric = "auc"
    )

###################################
###### GRID SEARCH ################
###################################
# https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
# https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/

# Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 100}

if grid_search:
    '''params = { 'max_depth': [4,5,6],
               'learning_rate': [0.01, 0.02],
               'n_estimators': [1000, 3000, 6000],
               'colsample_bytree': [0.5, 0.6, 0.7]}
    '''
    params = { 'reg_alpha': [0.3,1,5,10],
               'reg_lambda': [0.3,1,5,10],
           }
    clf = GridSearchCV(estimator=clf, 
                       param_grid=params,
                       scoring='neg_log_loss', 
                       verbose=4)
    
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
    
else:
    clf.fit(X_train, y_train, 
            eval_set              = [(X_train,y_train),(X_test, y_test)],
            early_stopping_rounds = 1000,
            eval_metric           = ['error'],
            #eval_metric           = ['auc','logloss'],
            verbose               = True,
            sample_weight         = train['w'],
        )


    ###########################
    ##### Learning Plots ######
    ###########################

    results = clf.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    
    '''# plot 
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    
    plt.ylabel('auc')
    plt.xlabel('epochs')
    plt.title('XGBoost auc')
    plt.savefig("learningcurve.png")
    '''

    ## plot classification error
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.savefig("learningcurve2.png")


    
    if not os.path.exists('bdtModel'):
        os.mkdir('bdtModel')        
    pickle.dump(clf, open('bdtModel/BDT_Model_' +flag+ '.pck', 'wb'))
    print('Model saved  ')
    pickle.dump(features, open('bdtModel/BDT_Model_' +flag+ '_features.pck', 'wb'))
    print('Features saved')


    print('enrich the data')
    for i, label in zip(range(2), ['bkg', 'tau']):
        tau['bdt_%s' %label] = clf.predict_proba(tau[features])[:,i]
        bkg['bdt_%s' %label] = clf.predict_proba(bkg[features])[:,i]
        data['bdt_%s' %label] = clf.predict_proba(data[features])[:,i]

    tau.to_root('tau_bdt.root', key='tree')
    bkg.to_root('bkg_bdt.root', key='tree')

    train_signal_pred = clf.predict_proba(train[train.target == 1][features])
    test_signal_pred = clf.predict_proba(test[test.target == 1][features])
    train_bkg_pred = clf.predict_proba(train[train.target == 0][features])
    test_bkg_pred = clf.predict_proba(test[test.target == 0][features])


    #####################################
    ###### Test Overfitting #############
    #####################################

    h1 = ROOT.TH1F("","",30,0,1)
    h2 = ROOT.TH1F("","",30,0,1)
    for t,b in zip(train_signal_pred,test_signal_pred):
        h1.Fill(t[1])
        h2.Fill(b[1])
    c1=ROOT.TCanvas()
    h1.Scale(1/h1.Integral())
    h2.Scale(1/h2.Integral())
    
    h1.Draw("hist")
    h2.SetLineColor(ROOT.kRed)
    h2.Draw("histSAME")
    c1.SaveAs("train_test_predicitons_sig.png")
    ks_score = h1.KolmogorovTest(h2)

    print("KS score: ",ks_score, len(train_signal_pred),len(test_signal_pred))


    h1 = ROOT.TH1F("","",30,0,1)
    h2 = ROOT.TH1F("","",30,0,1)
    for t,b in zip(train_bkg_pred,test_bkg_pred):
        h1.Fill(t[1])
        h2.Fill(b[1])
    c1=ROOT.TCanvas()
    h1.Scale(1/h1.Integral())
    h2.Scale(1/h2.Integral())
    
    h1.Draw("hist")
    h2.SetLineColor(ROOT.kRed)
    h2.Draw("histSAME")
    c1.SaveAs("train_test_predicitons_bkg.png")
    ks_score = h1.KolmogorovTest(h2)

    print("KS score: ",ks_score, len(train_bkg_pred),len(test_bkg_pred))


    #######################################
    #### ROC curve plotting ###############
    #######################################
    
    plt.clf()
    
    xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
    plt.plot(xy, xy, color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    fpr, tpr, wps = roc_curve(test['target'], test_pred.T[1])
    print("AUC test",sk.metrics.auc(fpr,tpr))
    fpr, tpr, wps = roc_curve(train['target'],train_pred.T[1])
    print("AUC train",sk.metrics.auc(fpr,tpr))
    fpr, tpr, wps = roc_curve(data.target, data.bdt_tau)
    
    
    plt.plot(fpr, tpr, label='bkg vs. tau', color='b')
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

    plt.legend()

    plt.savefig('rocs_bdt_%s.pdf' %flag)

