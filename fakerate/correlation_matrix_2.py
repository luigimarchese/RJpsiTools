from histos_nordf import histos
import ROOT
import seaborn as sns
from root_pandas import read_root, to_root
from datetime import datetime
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ROOT
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

import sklearn as sk
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.activations import softmax
from keras.constraints import unit_norm
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import RobustScaler
from itertools import product
from new_branches_pandas import to_define
from selections_for_fakerate import preprepreselection,triggerselection, etaselection
from histos_nordf import histos #histos file NO root dataframes
#from samples import sample_names
from samples import sample_names_explicit_jpsimother_compressed as sample_names

def correlation_matrix(features_tmp, passing):
    ##########################################################################################
    #####   CORRELATION MATRIX SIGNAL
    ##########################################################################################
    # Compute the correlation matrix for the signal

    plt.clf()

    features = [item for item in features_tmp]
    #features = features[30:]
    #features = ['m_miss_sq','jpsivtx_lxy_unc','jpsivtx_cos2D','Q_sq','pt_var','pt_miss_vec','pt_miss_scal','E_mu_star','E_mu_canc','DR_mu1mu2','jpsi_pt','Bmass','mu1pt','mu2pt','kpt','Bpt','Bpt_reco','bvtx_lxy_unc','dr12','dr23','dr13','jpsiK_mass']
    #features= ['Q_sq','pt_var','E_mu_star','DR_mu1mu2','jpsi_pt','kpt','Bmass','keta','dr12','jpsiK_mass','bvtx_chi2','bvtx_lxy','bvtx_svprob','m_miss_sq','jpsivtx_lxy_unc']
    #features = ['Bmass','mu1pt','mu2pt','kpt','mu1phi','Bpt','Bpt_reco','dr12','dr23','dr13','jpsiK_mass','m_miss_sq','jpsivtx_lxy_unc','bvtx_cos2D','jpsivtx_chi2','jpsivtx_log10_lxy_sig','bvtx_log10_lxy_sig','Q_sq','pt_var','pt_miss_vec','pt_miss_scal','jpsi_pt','E_mu_star','E_mu_canc','DR_mu1mu2','bvtx_lxy_unc','bvtx_lxy','k_dxy_sig','k_dz_sig','bvtx_chi2','mu1_dz_sig','abs_mu2_dz']

    passing_array = passing[features]

    corr = passing[features ].corr()

    # check if they have correlation
    final_features = [features[0]]
    for i in range(1,len(features)):
        correlation_flag = 0
        print("i",i,features[i])
        for j in range(len(final_features)):
            print("j",j,final_features[j],corr[features[i]][final_features[j]])
            if abs(corr[features[i]][final_features[j]])>0.6:
                print(features[i] + " and "+final_features[j]+" are correlated:",corr[features[i]][final_features[j]])
                correlation_flag = 1
                break
        if not correlation_flag ==1:
            print("writing ",features[i])
            final_features.append(features[i])
    print(final_features)
    corr = passing[final_features ].corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                    square=True, linewidths=.8, cbar_kws={"shrink": .8})

    # rotate axis labels
    g.set_xticklabels(final_features, rotation='vertical')
    g.set_yticklabels(final_features, rotation='horizontal')

    # plt.show()
    plt.title('linear correlation matrix - pass')
    plt.tight_layout()
    plt.savefig('ROC/corr_pass_2.png' )
    plt.clf()


if __name__ == "__main__":

    data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_nopresel_withpresel_v2_withnn_withidiso.root'
    prepreselection = preprepreselection + "&" +triggerselection +"&"+etaselection + "& Bpt_reco<80"
    data = read_root(data_path, 'BTo3Mu', where=prepreselection )
    data = to_define(data)
    main_df = data

    pass_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel<0.2'
    fail_id = '(k_mediumID<0.5) & (k_raw_db_corr_iso03_rel>0.2)'

    #main_df is already shuffled
    passing_tmp   = main_df.copy()

    features_tmp = ['Q_sq','m_miss_sq','Bpt','jpsivtx_log10_lxy_sig', 'abs_Beta','ip3d_sig_dcorr','bvtx_log10_svprob','bvtx_cos2D','k_dxy_sig','k_dz_sig']#,'abs_mu1_dxy']

#alpha
#['Q_sq', 'E_mu_star', 'E_mu_canc', 'jpsiK_mass', 'pt_miss_vec', 'bvtx_lxy_unc', 'DR_jpsimu', 'm_miss_sq', 'pt_miss_scal', 'Bpt_reco', 'Bmass', 'dr13', 'jpsivtx_lxy_unc', 'ip3d_e_corr_new', 'Bpt', 'jpsi_pt', 'mu1pt', 'DR_mu1mu2', 'dr12', 'dr23', 'pt_var', 'bvtx_lxy_unc_corr', 'bvtx_cos2D', 'mu2pt', 'kpt', 'k_dxy_sig', 'bvtx_lxy_sig', 'k_dz_sig', 'decay_time_pv_jpsi', 'abs_k_dxy', 'bvtx_log10_lxy_sig', 'm13', 'jpsivtx_lxy_unc_corr', 'mu2_dxy_sig', 'jpsivtx_cos2D', 'm23', 'abs_k_dz', 'nPV', 'jpsivtx_lxy_sig', 'bvtx_lxy_sig_corr', 'bvtx_log10_svprob', 'mu2_dz_sig', 'jpsivtx_log10_lxy_sig', 'bvtx_chi2', 'bvtx_svprob', 'm12', 'jpsi_mass', 'ip3d_e', 'bvtx_log10_lxy_sig_corr', 'abs_mu2_dxy', 'ip3d_corr', 'abs_Beta', 'abs_mu2_dz', 'jpsivtx_lxy_sig_corr', 'jpsivtx_log10_svprob', 'jpsivtx_svprob', 'mu1_dz_sig', 'jpsivtx_chi2', 'jpsivtx_log10_lxy_sig_corr']#, 'abs_mu1_dxy', 'jpsivtx_lxy', 'ip3d_sig_dcorr', 'bvtx_log10_lxy', 'keta', 'mu1eta', 'Beta', 'jpsi_eta', 'mu2eta', 'abs_mu1_dz', 'mu1phi', 'ip3d', 'jpsivtx_log10_lxy', 'Bphi', 'kphi', 'mu2phi', 'jpsi_phi', 'ip3d_sig', 'bvtx_lxy']

#mc
#['bvtx_log10_svprob', 'bvtx_chi2', 'bvtx_svprob', 'pt_var', 'ip3d', 'dr12', 'DR_mu1mu2', 'ip3d_corr', 'Q_sq', 'kpt', 'ip3d_e', 'ip3d_sig_dcorr', 'ip3d_sig', 'ip3d_e_corr_new', 'mu1pt', 'jpsi_pt', 'E_mu_star', 'mu2pt', 'jpsivtx_lxy_unc', 'jpsivtx_lxy_unc_corr', 'bvtx_cos2D', 'DR_jpsimu', 'dr13', 'abs_Beta', 'm_miss_sq', 'jpsivtx_cos2D', 'E_mu_canc', 'Bpt', 'jpsiK_mass', 'bvtx_lxy', 'dr23', 'bvtx_log10_lxy', 'bvtx_lxy_sig_corr', 'pt_miss_vec', 'bvtx_log10_lxy_sig_corr', 'bvtx_log10_lxy_sig', 'bvtx_lxy_sig', 'abs_k_dxy', 'k_dxy_sig', 'Bmass', 'm23', 'jpsivtx_lxy_sig_corr', 'jpsivtx_lxy_sig', 'pt_miss_scal', 'abs_mu1_dz', 'abs_mu2_dz', 'jpsivtx_log10_lxy_sig_corr', 'jpsivtx_log10_lxy_sig', 'mu2_dxy_sig', 'abs_mu1_dxy']

#data
#['Q_sq', 'E_mu_star', 'E_mu_canc', 'jpsiK_mass', 'pt_miss_vec', 'pt_miss_scal', 'm_miss_sq', 'pt_var', 'Bpt_reco', 'DR_mu1mu2', 'dr12', 'jpsi_pt', 'Bmass', 'mu1pt', 'DR_jpsimu', 'Bpt', 'dr13', 'bvtx_lxy_unc', 'bvtx_lxy_unc_corr', 'mu2pt', 'dr23', 'jpsivtx_lxy_unc', 'jpsivtx_lxy_unc_corr', 'decay_time_pv_jpsi', 'jpsivtx_cos2D', 'abs_k_dz', 'abs_Beta', 'bvtx_lxy_sig', 'bvtx_lxy_sig_corr', 'bvtx_log10_lxy_sig_corr', 'bvtx_log10_lxy_sig', 'm23', 'ip3d_e', 'ip3d_e_corr_new', 'k_dz_sig', 'abs_mu2_dz', 'abs_mu1_dz', 'jpsivtx_lxy_sig', 'jpsivtx_log10_lxy_sig_corr', 'jpsivtx_log10_lxy_sig', 'm13', 'abs_mu1_dxy', 'nPV', 'abs_mu2_dxy', 'kpt', 'abs_k_dxy', 'mu2_dz_sig', 'mu2_dxy_sig', 'ip3d_sig', 'ip3d_sig_dcorr', 'bvtx_log10_svprob', 'jpsivtx_lxy', 'jpsivtx_lxy_sig_corr', 'bvtx_chi2', 'bvtx_svprob', 'mu1_dz_sig', 'bvtx_lxy', 'k_dxy_sig', 'jpsivtx_log10_lxy', 'bvtx_log10_lxy', 'ip3d', 'ip3d_corr']#, 'bvtx_cos2D', 'jpsivtx_log10_svprob', 'jpsivtx_svprob', 'jpsivtx_chi2', 'keta', 'Beta', 'mu2eta', 'jpsi_eta', 'mu1eta', 'kphi', 'm12', 'jpsi_mass', 'jpsi_phi', 'Bphi', 'mu1phi', 'mu2phi']


    correlation_matrix(features_tmp, passing_tmp)
