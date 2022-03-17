'''
Train the BDT
'''
#https://github.com/rmanzoni/WTau3Mu/blob/92X/bdt/xgb_trainer_5_mar_2019.py#L254-L329

from scipy.stats import ks_2samp
import seaborn as sns
import ROOT
from itertools import product
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import svm
import xgboost as xgb
import os

from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from xgboost import plot_tree

import pandas as pd
import numpy as np
import pickle

from root_pandas import read_root, to_root
from new_branches import to_define 
from selections_for_bdt import preselection_fakes_1, preselection_fakes_2, preselection_mc
from datetime import datetime

# No need to retrain the BDT if it is already done
train_bdt = True
model_flag = '23Feb2022_10h18m04s' #used only if train_bdt = False

flag = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

print("Model and plots will be saved in bdt_models/%s"%flag)

features = [
    #'Q_sq'                 ,
    'mcorr'                ,
    #'pt_miss_vec'          ,
    'jpsivtx_log10_lxy'    ,
    'ip3d_sig_dcorr'      ,
    'mu2_dz_sig',
    'mu2_dxy_sig',
    'bvtx_svprob',
    #IMP MU2 e k
    #vtx prob
]
'''
features = [
    'pt_miss_vec',
    'mu2_dz_sig',
    'mu2_dxy_sig',
    'k_dz_sig',
    'k_dxy_sig',
    'mmm_p4_perp',
    'mcorr',
    'jpsivtx_log10_lxy_sig',
    'ip3d_sig_dcorr',
]
'''

if not os.path.exists('bdt_models/%s/'%flag):
    os.mkdir('bdt_models/%s/'%flag)        

classifier_file = open('bdt_models/%s/classifiers_%s.pck' % (flag,flag), 'wb')
samples = dict()

tree_name = 'BTo3Mu'
#tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021'

samples['tau'] = ROOT.RDataFrame(tree_name, '%s/jpsi_tau_bdt_vv1.root'     %tree_dir)
samples['fakes'] = ROOT.RDataFrame(tree_name, '%s/data_bdt_vv1.root'    %tree_dir)  

print('Adding new columns...')
for k, v in samples.items():
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column): continue
        samples[k] = samples[k].Define(new_column, new_definition)

print("Selections...")
tau = pd.DataFrame(samples['tau'].Filter(preselection_mc).AsNumpy())
fakes1 = pd.DataFrame(samples['fakes'].Filter(preselection_fakes_1).AsNumpy())  # doesn't pass the id and pass iso
fakes2 = pd.DataFrame(samples['fakes'].Filter(preselection_fakes_2).AsNumpy())  # doesn't pass the id and do not pass iso

print('Defining targets...')
tau['target'] = np.ones ( tau.shape[0]).astype(int)
fakes1['target'] = np.zeros( fakes1.shape[0]).astype(int)
fakes2['target'] = np.zeros( fakes2.shape[0]).astype(int)

fakes1['nn_weight'] = np.ones(fakes1.shape[0]).astype(int)
fakes2['nn_weight'] = fakes2['fakerate_data_2']

fakes = pd.concat([fakes1,fakes2])

minn = min(tau.shape[0],fakes.shape[0])
tau['w_number'] = np.ones(tau.shape[0]) * minn/tau.shape[0]
tau['w'] = tau['w_number']
fakes['w_number'] = np.ones(fakes.shape[0]) * minn/fakes.shape[0]
fakes['w'] = fakes['w_number']* fakes['nn_weight']
print(len(tau),len(fakes),tau['w'],fakes['w'])

# concatenate the bkg and signal
sample = pd.concat([fakes, tau])
sample = sample.reset_index()

train, test = train_test_split(sample, test_size=0.2, random_state=1986)
train, valid = train_test_split(train, test_size=0.2, random_state=1986)

X_train, X_valid = train[features], valid[features]
y_train, y_valid = train['target'], valid['target']
weight = train['w']
weight_val = valid['w']

clf = xgb.XGBClassifier(
    max_depth        = 3,
    learning_rate    = 0.01, 
    n_estimators     = 1000, 
    #silent           = False,
    subsample        = 0.6,
    colsample_bytree = 0.7,
    min_child_weight = 4E-06,
    #gamma            = 0.1, 
    seed             = 1986,)

clf.fit(X_train, y_train, 
        eval_set              = [(X_train,y_train),(X_valid, y_valid)],
        #early_stopping_rounds = 50,
        eval_metric           = ['auc'],
        #verbose               = True,
        sample_weight         = weight,)


###################
### Predictions ###
###################
sub = pd.DataFrame()
sub['id']     = np.array(test.index)
sub['target'] = np.array(test.target)

# Predict on our test data
p_test = clf.predict_proba(test[features])[:, 1]
sub['score'] = p_test

# adjust the score to match 0,1
smin = min(p_test)
smax = max(p_test)

sub['score_norm'] = (p_test - smin) / (smax - smin)

print ('\tcross validation error      %.5f' %(np.sum(np.abs(sub['score_norm'] - sub['target']))/len(sub)))
print ('\tcross validation signal     %.5f' %(np.sum(np.abs(sub[sub.target>0.5]['score_norm'] - sub[sub.target>0.5]['target']))/len(sub)))
print ('\tcross validation background %.5f' %(np.sum(np.abs(sub[sub.target<0.5]['score_norm'] - sub[sub.target<0.5]['target']))/len(sub)))

pickle.dump(clf, classifier_file)
classifier_file.close()
pickle.dump(features, open('bdt_models/%s/features_'%flag +flag+ '.pck', 'wb'))

##########################################################################################

train = train.copy()
test = test.copy()
tau = tau.copy()
fakes = fakes.copy()

train.loc[:,'bdt'] = np.zeros_like(train.mez_min)
test.loc[:,'bdt'] = np.zeros_like(test .mez_min)
tau.loc[:,'bdt'] = np.zeros_like(tau.mez_min)
fakes.loc[:,'bdt'] = np.zeros_like(fakes.mez_min)

train['bdt'] += clf.predict_proba(train[features])[:, 1]
test['bdt'] += clf.predict_proba(test [features])[:, 1]
tau['bdt'] += clf.predict_proba(tau[features])[:, 1] 
fakes  ['bdt'] += clf.predict_proba(fakes[features])[:, 1]

test.to_root("prova.root")
##########################################################################################
#####   ROC CURVE
##########################################################################################
plt.clf()

#cuts_to_display = [0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.7,0.75,0.80, 0.82, 0.84, 0.86, 0.89]
cuts_to_display = [0.40,0.45,0.5,0.55,0.6,0.7,0.75,0.80, 0.82, 0.84, 0.86, 0.89]
#cuts_to_display = [0.40,0.45,0.5,0.55,0.6,0.7,0.75]#,0.80, 0.82, 0.84, 0.86, 0.89]

xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
plt.plot(xy, xy, color='grey', linestyle='--')
plt.xlim([10**-2, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xscale('log')

fpr, tpr, wps = roc_curve(test.target, test.bdt, sample_weight=test.w)
plt.plot(fpr, tpr, label='test sample', color='b')

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps>icut).sum()
    wp_x.append(fpr[idx])
    wp_y.append(tpr[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

fpr, tpr, wps = roc_curve(train.target, train.bdt, sample_weight=train.w)
plt.plot(fpr, tpr, label='train sample', color='r')

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps>icut).sum()
    wp_x.append(fpr[idx])
    wp_y.append(tpr[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

print ('ROC AUC train ', roc_auc_score(train.target, train.bdt, sample_weight=train.w))
print ('ROC AUC test  ', roc_auc_score(test.target , test.bdt , sample_weight=test.w))

plt.legend(loc='best')
plt.grid()
plt.title('ROC')
#plt.tight_layout()
#plt.savefig('bdt_models/%s/roc_%s.png' %(flag,flag))
#plt.clf()
#print("ROC SAVED.")
#roc_file = open('roc_%s.pck' % flag, 'w+')
#pickle.dump((tpr, fpr), roc_file)
#roc_file.close()

#plt.clf()

#cuts_to_display = [0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.7,0.75,0.80, 0.82, 0.84, 0.86, 0.89]
cuts_to_display = [-2,-1.5,-1,-0.5,0, 0.5,1,1.5,2]
#cuts_to_display = [0.40,0.45,0.5,0.55,0.6,0.7,0.75]#,0.80, 0.82, 0.84, 0.86, 0.89]

#xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
#plt.plot(xy, xy, color='grey', linestyle='--')
#plt.xlim([10**-5, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')

#plt.xscale('log')

fpr, tpr, wps = roc_curve(sample.run==1, sample.ip3d_sig_dcorr)
plt.plot(fpr, tpr, label='ip3d', color='g')

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps>icut).sum()
    wp_x.append(fpr[idx])
    wp_y.append(tpr[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

#print ('ROC AUC sample ', roc_auc_score(sample.target, sample.bdt, sample_weight=sample.w))

plt.legend(loc='best')
plt.grid()
plt.title('ROC')
#plt.tight_layout()
plt.savefig('bdt_models/%s/roc_ip3d_%s.png' %(flag,flag))
plt.clf()
print("ROC SAVED.")
#roc_file = open('roc_%s.pck' % flag, 'w+')
#pickle.dump((tpr, fpr), roc_file)
#roc_file.close()

##########################################################################################
#####   OVERTRAINING TEST
##########################################################################################
train_sig = train[train.target>0.5].bdt
train_bkg = train[train.target<0.5].bdt

test_sig = test[test.target>0.5].bdt
test_bkg = test[test.target<0.5].bdt

low  = 0
high = 1
low_high = (low,high)
bins = 50


#################################################
hist, bins = np.histogram(
    test_sig,
    bins=bins, 
    range=low_high, 
    density=True
)

width  = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
scale  = len(test_sig) / sum(hist)
err    = np.sqrt(hist * scale) / scale

plt.errorbar(
    center, 
    hist, 
    yerr=err, 
    fmt='o', 
    c='r', 
    label='S (test)'
)

#################################################
sns.distplot(train_sig, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "r"}, label='S (train)')

#################################################
hist, bins = np.histogram(
    test_bkg,
    bins=bins, 
#     range=low_high, 
    density=True
)

width  = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
scale  = len(test_bkg) / sum(hist)
err    = np.sqrt(hist * scale) / scale

plt.errorbar(
    center, 
    hist, 
    yerr=err, 
    fmt='o', 
    c='b', 
    label='B (test)'
)

#################################################
sns.distplot(train_bkg, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "b"}, label='B (train)')

#################################################
plt.xlabel('BDT output')
plt.ylabel('Arbitrary units')
plt.legend(loc='best')
ks_sig = ks_2samp(train_sig, test_sig)
ks_bkg = ks_2samp(train_bkg, test_bkg)
plt.suptitle('KS p-value: sig = %.3f%s - bkg = %.2f%s' %(ks_sig.pvalue * 100., '%', ks_bkg.pvalue * 100., '%'))

# train_sig_w = np.ones_like(train_sig) * 1./len(train_sig)
# train_bkg_w = np.ones_like(train_bkg) * 1./len(train_bkg)
# test_sig_w  = np.ones_like(test_sig)  * 1./len(test_sig )
# test_bkg_w  = np.ones_like(test_bkg)  * 1./len(test_bkg )
# 
# ks_sig = ks_w2(train_sig, test_sig, train_sig_w, test_sig_w)
# ks_bkg = ks_w2(train_bkg, test_bkg, train_bkg_w, test_bkg_w)
# plt.suptitle('KS p-value: sig = %.3f%s - bkg = %.2f%s' %(ks_sig * 100., '%', ks_bkg * 100., '%'))

plt.savefig('bdt_models/%s/overtrain_%s.png' %(flag,flag))

plt.yscale('log')

plt.savefig('bdt_models/%s/overtrain_log_%s.png' %(flag,flag))

plt.clf()

print("Overtraining test saved.")
# no K fold

##########################################################################################
#####   FEATURE IMPORTANCE
##########################################################################################
fscores = OrderedDict(zip(features, np.zeros(len(features))))

myscores = clf.get_booster().get_fscore()
for jj in myscores.keys():
    fscores[jj] += myscores[jj]

#print(fscores, fscores.values(), list(fscores.values()),np.sum(fscores.values()))
totalsplits = np.sum(list(fscores.values()))

for k, v in fscores.items():
    print(k,v,totalsplits)
    fscores[k] = v/totalsplits 

plt.xlabel('relative F-score')
plt.ylabel('feature')

orderedfscores = OrderedDict(sorted(fscores.items(), key=lambda x : x[1], reverse=False ))

bars = features
y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, orderedfscores.values())
 
# Create names on the y-axis
plt.yticks(y_pos, bars)

# plot_importance(clf)
#plt.tight_layout()
plt.savefig('bdt_models/%s/feat_importance_%s.png' %(flag,flag))
print('bdt_models/%s/feat_importance_%s.png' %(flag,flag) + ' saved.')
plt.clf()


##########################################################################################
#####   CORRELATION MATRIX SIGNAL
##########################################################################################
# Compute the correlation matrix for the signal
corr = tau[features + ['bdt']].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                square=True, linewidths=.8, cbar_kws={"shrink": .8})

# rotate axis labels
g.set_xticklabels(features+['bdt'], rotation='vertical')
g.set_yticklabels(features+['bdt'], rotation='horizontal')

# plt.show()
plt.title('linear correlation matrix - signal')
plt.tight_layout()
plt.savefig('bdt_models/%s/corr_sig_%s.png' %(flag,flag))
print('bdt_models/%s/corr_sig_%s.png' %(flag,flag)+' saved')
plt.clf()

##########################################################################################
#####   CORRELATION MATRIX BACKGROUND
##########################################################################################
# Compute the correlation matrix for the signal
corr = fakes[features + ['bdt']].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                square=True, linewidths=.8, cbar_kws={"shrink": .8})

# rotate axis labels
g.set_xticklabels(features+['bdt'], rotation='vertical')
g.set_yticklabels(features+['bdt'], rotation='horizontal')

# plt.show()
plt.title('linear correlation matrix - background')
plt.tight_layout()
plt.savefig('bdt_models/%s/corr_bkg_%s.png' %(flag,flag))
print('bdt_models/%s/corr_bkg_%s.png' %(flag,flag)+' saved')

