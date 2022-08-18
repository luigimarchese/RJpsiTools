
'''
Script that computes the NN to find dependencies of fr to other variables
Also performs the closure test
- train with data and MC
'''

import sklearn as sk
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

epochs = 50
#no pop-up windows
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
officialStyle(ROOT.gStyle, ROOT.TGaxis)

def preprocessing(passing, failing):
    '''
    Preprocessing of data before training/testing the NN
    '''
    print("Preprocessing")

    # concatenate the events and shuffle
    main_df = pd.concat([passing, failing], sort=False)
    #reindex
    main_df.index = np.array(range(len(main_df)))
    main_df = main_df.sample(frac=1, replace=False, random_state=1986) # shuffle

    # X and Y
    X = pd.DataFrame(main_df, columns=list(set(features)))
    Y = pd.DataFrame(main_df, columns=['target'])
    #norm
    xx,qt = norm(X)
    pickle.dump( qt, open( '/'.join([final_nn_path, 'input_tranformation_weighted.pck']), 'wb' ) )
    return xx,qt,Y,main_df

def preprocessing_whole_sample(df):
    '''
    Preprocessing of data before training/testing the NN
    '''
    print("Preprocessing")

    # concatenate the events and shuffle
    main_df = df
    #reindex
    main_df.index = np.array(range(len(main_df)))
    main_df = main_df.sample(frac=1, replace=False, random_state=1986) # shuffle

    # X and Y
    X = pd.DataFrame(main_df, columns=list(set(features)))
    Y = pd.DataFrame(main_df, columns=['target'])
    #norm
    xx,qt = norm(X)
    pickle.dump( qt, open( '/'.join([final_nn_path, 'input_tranformation_weighted.pck']), 'wb' ) )
    return xx,qt,Y,main_df

def preprocessing_just_scale(passing, failing, qt):
    '''
    Preprocessing of data before training/testing the NN
    '''
    print("RESCALE FEATURES")
    main_df = pd.concat([passing, failing], sort=False)
    #reindex
    main_df.index = np.array(range(len(main_df)))
    main_df = main_df.sample(frac=1, replace=False, random_state=1986) # shuffle
    
    sample_weight = pd.DataFrame(main_df, columns=['sample_weight'])
    X = pd.DataFrame(main_df, columns=list(set(features)))
    Y = pd.DataFrame(main_df, columns=['target'])
    xx = qt.transform(X[features])

    return xx,Y,sample_weight

def norm(X):
    '''
    Preprocessing of data 
    '''
    print("StndScaler   ")
    #norm
    qt = RobustScaler()
    qt.fit(X[features])
    xx = qt.transform(X[features])
    return xx,qt


def closure_test(passing_mc_ct,failing_mc_ct):
    '''
    Histos for the closure test are created here.
    3 histos made with MC HbToJpsiMuMU with preselection and cut on third muon applied
    1. pass
    2. fail
    3. fail with NN weights taken from the efficiency computation -> should be equal in shape and yield to pass
    '''
    print("#########################################")
    print("####        Closure Test             ####")
    print("#########################################")
    for var in histos:
        print("Computing now variable "+ var)
        
        #histo for the MC in the pass region
        hist_pass = ROOT.TH1D("pass"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in passing_mc_ct[histos[var][0]]:
            hist_pass.Fill(item)

        # histo for the MC in the fail reigon
        hist_fail = ROOT.TH1D("fail"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in failing_mc_ct[histos[var][0]]:
            hist_fail.Fill(item)

        hist_pass_w = ROOT.TH1D("passw"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item,nn in zip(failing_mc_ct[histos[var][0]],failing_mc_ct['nn']):
            hist_pass_w.Fill(item,nn/(1-nn))

        c1.cd()
        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)
        main_pad.cd()
        main_pad.SetLogy(False)

        hist_pass.GetXaxis().SetTitle(histos[var][5])
        hist_pass.GetYaxis().SetTitle('events')
        hist_pass.SetLineColor(ROOT.kMagenta)
        hist_pass.SetFillColor(0)

        hist_pass_w.GetXaxis().SetTitle(histos[var][5])
        hist_pass_w.GetYaxis().SetTitle('events')
        hist_pass_w.SetLineColor(ROOT.kOrange)
        hist_pass_w.SetFillColor(0)

        hist_fail.GetXaxis().SetTitle(histos[var][5])
        hist_fail.GetYaxis().SetTitle('events')
        hist_fail.SetLineColor(ROOT.kBlue)
        hist_fail.SetFillColor(0)

        maximum = max(hist_pass.GetMaximum(),hist_fail.GetMaximum(),hist_pass_w.GetMaximum()) 
        hist_pass.SetMaximum(2.*maximum)

        CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

        hist_pass.Draw('hist ')
        hist_fail.Draw('hist same')
        hist_pass_w.Draw('hist same')

        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        leg.AddEntry(hist_pass, 'hb_pass','F')
        leg.AddEntry(hist_fail, 'hb_fail','F')
        leg.AddEntry(hist_pass_w, 'hb_fail_nn','F')

        leg.Draw('same')

        # Kilmogorov test
        h_pass = hist_pass.Clone("h_pass")
        h_fail = hist_fail.Clone("h_fail")
        h_pass_w = hist_pass_w.Clone("h_pass_w")
        h_pass.Scale(1./h_pass.Integral())
        h_pass_w.Scale(1./h_pass_w.Integral())
        h_fail.Scale(1./h_fail.Integral())
        h_ratio_passpass = h_pass.Clone("h_ratio_passpass")
        h_ratio_passpass.Divide(h_pass,h_pass)
        h_ratio_failpass = h_fail.Clone("h_ratio_failpass")
        h_ratio_failpass.Divide(h_fail,h_pass)
        h_ratio_passwpass = h_pass_w.Clone("h_ratio_passwpass")
        h_ratio_passwpass.Divide(h_pass_w,h_pass)

        ks_fail = h_ratio_failpass.KolmogorovTest(h_ratio_passpass)
        ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
        print("KS fail: ",ks_fail)
        print("KS fail rw: ",ks_fail_rw)

        KS_value = ROOT.TPaveText(0.66, 0.7, 0.92, 0.8, 'nbNDC')
        KS_value.AddText('KS fail    = %.4f' %ks_fail)
        KS_value.AddText('KS fail rw = %.4f' %ks_fail_rw)
        KS_value.SetFillColor(0)
        KS_value.Draw('EP')

        #c1.Modified()
        #c1.Update()
        #c1.SaveAs(final_nn_path + '/closure_test/%s.pdf' %(var))
        c1.SaveAs(final_nn_path + '/closure_test/%s.png' %( var))

        maximum = max(h_pass.GetMaximum(),h_fail.GetMaximum(),h_pass_w.GetMaximum()) 
        h_pass.SetMaximum(2.*maximum)
        h_pass.Draw('hist ')
        h_fail.Draw('hist same')
        h_pass_w.Draw('hist same')
        leg.Draw('same')
        KS_value.Draw('EP')
        c1.SaveAs(final_nn_path + '/closure_test/norm/%s.png' %( var))

if __name__ == "__main__":
        
    #Hb to jpsi X MC sample

    #data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_fakerate_only_iso.root'
    mc_path = []
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_tau_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic0_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic1_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic2_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_hc_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/hc_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/psi2s_mu_nopresel_withpresel_v2.root')
    mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/psi2s_tau_nopresel_withpresel_v2.root')
    jpsix_path = []
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bzero_nopresel_withpresel_v2.root')
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bplus_nopresel_withpresel_v2.root')
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bzero_s_nopresel_withpresel_v2.root')
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_sigma_nopresel_withpresel_v2.root')
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_lambdazero_b_nopresel_withpresel_v2.root')
    jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_xi_nopresel_withpresel_v2.root')
    
    #output path
    nn_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/nn/'
    
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    final_nn_path = nn_path + label 
    print("label:",label)
    features = [
        #'ptcone_k',
        #'keta',
        #'decay_time_pv_jpsi',
        #'Bpt_reco_log',
        #'Bpt',
        #'abs_Beta',
        'Q_sq',
        #'nPV',
        #'jpsivtx_log10_lxy_sig',
        #'Bmass',
        #'Beta',
        #'mu1eta',
        #'Beta',
        #'Bmass',
        #'nPV',
        #'kpt',
        #'abs_k_eta',
        #'abs_mu1_eta',
        #'abs_mu2_eta',
        #'kphi',
        #'kmass',
        #'mu1pt',
        #'mu1eta',
        #'mu1phi',
        #'mu1mass',
        #'mu2pt',
        #'mu2eta',
        #'mu2mass',
        #'mu2phi',
        #'Bmass',
        #'jpsiK_mass',
        #'is_mc',
        #'is_mc_jpsix',
    ]
    
    os.system('mkdir -p '+ final_nn_path + '/model/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/norm/')
    
    print("Hello!")
    
    prepreselection = preprepreselection + "&" +triggerselection +"&"+etaselection + "& Bpt_reco<80"

    mc = []
    for mc_p in mc_path:
        print(mc_p)
        mcc=read_root(mc_p, 'BTo3Mu', where=prepreselection + " & (abs(k_genpdgId)==13)")
        #mc.index = np.array(range(len(mc)))
        mcc = to_define(mcc)
        mcc['is_mc']= [1 for i in range(len(mcc.Bmass))]
        mcc['is_mc_jpsix']= [0 for i in range(len(mcc.Bmass))]
        mcc['sample_weight']=[-1 for i in range(len(mcc.Bmass))]
        mc.append(mcc)

    mcjpsix = []
    for mc_p in jpsix_path:
        mcc=read_root(mc_p, 'BTo3Mu', where=prepreselection + " & (abs(k_genpdgId)==13)")
        #mc.index = np.array(range(len(mc)))
        mcc = to_define(mcc)
        mcc['is_mc_jpsix']= [1 for i in range(len(mcc.Bmass))]
        mcc['is_mc']= [0 for i in range(len(mcc.Bmass))]
        mcc['sample_weight']=[-1 for i in range(len(mcc.Bmass))]
        mcjpsix.append(mcc)
    

    mc = pd.concat(mc, ignore_index=True)
    print("len mc",len(mc.Bmass))
    mcjpsix = pd.concat(mcjpsix, ignore_index=True)
    print("len mc jpsix",len(mcjpsix.Bmass))
    mc['w']   = 0.09 *1.1 *1.04 * 0.85 * 0.9 *1.4
    mcjpsix['w'] = 0.3 * 0.85 *0.7*0.1 * 2.7 *1.6 *0.85 * 1.8 *1.4 * mcjpsix['jpsimother_weight']

    data = pd.concat([mc,mcjpsix], ignore_index=True)
    
    print("len data",len(data.Bmass))

    # In this version I find the mena and variance for the scaler before going into the not mediumID region
    #print("data before preprocessing",data)
    #xx, qt, Y, main_df = preprocessing_whole_sample(data)
    #print("data after preprocessing",main_df)
    #print("MEAN OF THE FEATURES",qt.center_,qt.scale_)

    pass_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel<0.2' 
    fail_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel>0.2'

    #main_df is already shuffled
    passing_tmp   = data.query(pass_id).copy()
    failing_tmp   = data.query(fail_id).copy()

    print(len(passing_tmp.Bmass),len(failing_tmp.Bmass))
    passing_tmp.loc[:,'target'] = np.ones (passing_tmp.shape[0]).astype(int)
    failing_tmp.loc[:,'target'] = np.zeros (failing_tmp.shape[0]).astype(int)
    
    #reindexing to know which one I used for training, so the rest I use for the closure test
    passing_tmp.index = np.array(range(len(passing_tmp)))
    failing_tmp.index = np.array(range(len(failing_tmp)))
    
    passing = passing_tmp.sample(frac = 0.85, replace=False, random_state=1986)
    failing = failing_tmp.sample(frac = 0.85, replace=False, random_state=1986)
    
    #samples for the closure test
    passing_ct = passing_tmp.drop(passing.index)
    failing_ct = failing_tmp.drop(failing.index)
    #print(len(passing_ct.Q_sq),len(failing_ct.Q_sq))
    #print(passing,failing)

    xx, qt, Y, main_df = preprocessing(passing,failing)
    '''
    print("##########################")
    print("########  Scatter plot  #########")
    print("##########################")

    c1=ROOT.TCanvas()
    h_pre = ROOT.TH2F("","",100,1,2,100,0,11)
    for f0,f1 in zip(main_df[features[0]],main_df[features[1]]):
        h_pre.Fill(f0,f1)
    print(h_pre.Integral())
    c1.Draw()
    h_pre.SetTitle("Scatter Plot 2D Pre Scaler;%s;%s"%(features[0],features[1]))
    h_pre.Draw("coltz")

    c1.SaveAs(final_nn_path+'/model/scatter2d_pre.png')

    h_post = ROOT.TH2F("","",100,-2,2,100,-2,2)
    for item in xx:
        h_post.Fill(item[0],item[1])
    
    c1.Draw()
    h_post.SetTitle("Scatter Plot 2D Post Scaler;%s;%s"%(features[0],features[1]))
    h_post.Draw("coltz")
    print(final_nn_path)
    c1.SaveAs(final_nn_path+ '/model/scatter2d_post.png')
    '''
    print("##########################")
    print("########  Model  #########")
    print("##########################")
    
    activation = 'relu'
    
    # define the net
    input  = Input((len(features),))
    layer1  = Dense(2, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
    layer  = Dropout(0.1, name='dropout1')(layer1)
    #    layer3  = BatchNormalization(name='bn1')(layer2)
    #layer3  = Dense(2, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer2)
    #layer  = Dropout(0.1, name='dropout2')(layer3)
    #layer  = Dense(, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer3)
    #layer  = Dense(32, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
    output = Dense(  1, activation='sigmoid', name='output', )(layer)
    
    # Define outputs of your model
    model = Model(input, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'acc'])
    
    print(model.summary())
    
    # plot the models
    # https://keras.io/visualization/
    #plot_model(model, show_shapes=True, show_layer_names=True, to_file='/'.join([final_nn_path, '/model/model.png']) )
    
    # save the exact list of features
    pickle.dump( features, open( '/'.join([final_nn_path, '/model/input_features.pck']), 'wb' ) )
    
    # early stopping
    monitor = 'val_loss'
    es = EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=50, restore_best_weights=True)
    
    # reduce learning rate when at plateau, fine search the minimum
    reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='auto', factor=0.2, patience=5, min_lr=0.00001, cooldown=10, verbose=True)
    
    # save the model every now and then
    filepath = '/'.join([final_nn_path, '/model/saved-model-{epoch:04d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.h5'])
    save_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    # train only the classifier. beta is set at 0 and the discriminator is not trained
    callbacks = [reduce_lr, save_model]
    
    x_train_tmp, x_test, y_train, y_test = train_test_split(main_df, Y, test_size=0.2, shuffle= True)

    x_test_pass = qt.transform(x_test.query(pass_id)[features])
    x_test_pass = qt.transform(x_test_pass)
    x_test_fail = qt.transform(x_test.query(fail_id)[features])
    x_test_fail = qt.transform(x_test_fail)
    print("x_pass_test shape",x_test_pass.shape,x_test_fail.shape)
    x_test = pd.DataFrame(x_test, columns=list(set(features)))
    x_test = qt.transform(x_test)

    x_train_tmp, x_val_tmp, y_train, y_val = train_test_split(x_train_tmp, y_train, test_size=0.2, shuffle= True)
    
    x_train_pass = qt.transform(x_train_tmp.query(pass_id)[features])
    x_train_pass = qt.transform(x_train_pass)
    x_train_fail = qt.transform(x_train_tmp.query(fail_id)[features])
    x_train_fail = qt.transform(x_train_fail)
    x_train = pd.DataFrame(x_train_tmp, columns=list(set(features)))
    x_train = qt.transform(x_train)

    x_val = pd.DataFrame(x_val_tmp, columns=list(set(features)))
    x_val = qt.transform(x_val)
    #x_train, x_test, y_train, y_test = train_test_split(xx, Y, test_size=0.2, shuffle= True)

    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)

    for epoch in range(epochs):
        history = model.fit(x_train, y_train, sample_weight = x_train_tmp['w'], validation_data=(x_val, y_val, x_val_tmp['w']),epochs=1, batch_size=32, verbose=True)  

        ####################################
        ###### ROC CURVE ###################
        ####################################
        train_pred = model.predict(x_train)
        train_pred = [i[0] for i in train_pred]
        # let sklearn do the heavy lifting and compute the ROC curves for you
        print(y_train)
        print("train",train_pred)
        fpr, tpr, wps = roc_curve(y_train['target'], train_pred) 
        plt.plot(fpr, tpr, label='train ROC')
        print("AUC train",sk.metrics.auc(fpr,tpr))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        
        test_pred = model.predict(x_test)
        test_pred = [i[0] for i in test_pred]
        fpr, tpr, wps = roc_curve(y_test['target'], test_pred) 
        plt.plot(fpr, tpr, label='test ROC')
        print("AUC test",sk.metrics.auc(fpr,tpr))
        
        xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
        plt.plot(xy, xy, color='grey', linestyle='--')
        plt.yscale('linear')
        plt.legend()
        plt.savefig('/'.join([final_nn_path, '/model/roc_weighted_'+str(epoch)+'.pdf']) )
        print("RoC cuerve saved")
        
        plt.clf()
        
        #xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
        #plt.plot(xy, xy, color='grey', linestyle='--')
        
        # model.save_weights('net_model_weights.h5')
        
        ###################################
        ####### TEST overfitting ##########
        ###################################
        

        train_pred_pass = model.predict(x_train_pass)
        test_pred_pass = model.predict(x_test_pass)
        
        #plot
        h1 = ROOT.TH1F("","train",30,0,1)
        h2 = ROOT.TH1F("","test",30,0,1)
        for t,b in zip(train_pred_pass,test_pred_pass):
            h1.Fill(t)
            h2.Fill(b)
        c1=ROOT.TCanvas()
        h1.Scale(1/h1.Integral())
        h2.Scale(1/h2.Integral())
        c1.Draw()
        h1.Draw("hist")
        h2.SetLineColor(ROOT.kRed)
        h2.SetFillColor(ROOT.kWhite)
        h1.SetFillColor(ROOT.kWhite)
        h2.Draw("hist SAME")
        c1.BuildLegend()
        ks_score = h1.KolmogorovTest(h2)
        ks_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
        ks_value.AddText('ks score pass = %.4f' %ks_score)
        ks_value.SetFillColor(0)
        ks_value.Draw('EP same')
        
        c1.SaveAs(final_nn_path + "/model/KS_overfitting_test_pass_"+str(epoch)+".png")
        
        #ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
        print("KS score: ",ks_score, len(train_pred),len(test_pred))
        #print("KS fail rw: ",ks_fail_rw)
        
        train_pred_fail = model.predict(x_train_fail)
        test_pred_fail = model.predict(x_test_fail)
        
        #plot
        h1 = ROOT.TH1F("","train",30,0,1)
        h2 = ROOT.TH1F("","test",30,0,1)
        for t,b in zip(train_pred_fail,test_pred_fail):
            h1.Fill(t)
            h2.Fill(b)
        c1=ROOT.TCanvas()
        h1.Scale(1/h1.Integral())
        h2.Scale(1/h2.Integral())
        c1.Draw()
        h1.Draw("hist")
        h2.SetLineColor(ROOT.kRed)
        h2.SetFillColor(ROOT.kWhite)
        h1.SetFillColor(ROOT.kWhite)
        h2.Draw("hist SAME")
        c1.BuildLegend()
        ks_score = h1.KolmogorovTest(h2)
        ks_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
        ks_value.AddText('ks score fail = %.4f' %ks_score)
        ks_value.SetFillColor(0)
        ks_value.Draw('EP same')
        
        c1.SaveAs(final_nn_path + "/model/KS_overfitting_test_fail_"+str(epoch)+".png")
        
        #ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
        print("KS score: ",ks_score, len(train_pred),len(test_pred))
        #print("KS fail rw: ",ks_fail_rw)

        # save model and weights
        model.save('/'.join([final_nn_path, '/model/net_model_weighted_'+str(epoch)+'.h5']) )


    ##########################################################################################
    #####   CORRELATION MATRIX SIGNAL
    ##########################################################################################
    # Compute the correlation matrix for the signal

    passing_array = qt.transform(passing[features])
    failing_array = qt.transform(failing[features])
    print(passing_array)
    print(len(passing_array))
    print(passing.shape)
    print(model.predict(passing_array))
    print(passing)
    #print([i[0] for i in model.predict(passing_array)])
    passing.loc[:,'nn'] = [i[0] for i in model.predict(passing_array)]
    failing.loc[:,'nn'] = [i[0] for i in model.predict(failing_array)]

    corr = passing[features + ['nn']].corr()
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                    square=True, linewidths=.8, cbar_kws={"shrink": .8})

    # rotate axis labels
    g.set_xticklabels(features+['nn'], rotation='vertical')
    g.set_yticklabels(features+['nn'], rotation='horizontal')

    # plt.show()
    plt.title('linear correlation matrix - pass')
    plt.tight_layout()
    plt.savefig(final_nn_path + '/model/corr_pass.png' )
    plt.clf()

    ##########################################################################################
    #####   CORRELATION MATRIX BACKGROUND
    ##########################################################################################
    # Compute the correlation matrix for the signal
    corr = failing[features + ['nn']].corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                    square=True, linewidths=.8, cbar_kws={"shrink": .8})
    
    # rotate axis labels
    g.set_xticklabels(features+['nn'], rotation='vertical')
    g.set_yticklabels(features+['nn'], rotation='horizontal')

    # plt.show()
    plt.title('linear correlation matrix - fail')
    plt.tight_layout()
    plt.savefig(final_nn_path + '/model/corr_fail.png' )


    print("###########################################")
    print("########  Loss Plot  #######")
    print("###########################################")
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    epochs_range = range(1,epochs+1)
    epochs = epochs_range
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(final_nn_path + "/model/plotloss.png")    

    plt.clf()

    loss_train = history.history['acc']
    loss_val = history.history['val_acc']
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(final_nn_path + "/model/plotacc.png")    

    print("################################")
    print("########   Closure Test  #######")
    print("################################")
    ###### Closure test #######

    x_clos, qt_clos, y_clos, main_df_ct = preprocessing(passing_ct,failing_ct)
    
    main_df_ct.loc[:,'nn'] = model.predict(x_clos)
    
    c1 = ROOT.TCanvas('c1', '', 700, 700)
    c1.Draw()
    c1.cd()
    main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
    main_pad.Draw()
    main_pad.SetTicks(True)
    main_pad.SetBottomMargin(0.2)
    
    closure_test(main_df_ct[main_df_ct.target == 1], main_df_ct[main_df_ct.target == 0])


    print("###########################################")
    print("########  Add NN branch in samples  #######")
    print("###########################################")
    '''
    # open all samples as pandas
    samples = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    #tree_dir_dec2021 = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/' # same samples as Oct21, but new NN for fakerate
    tree_dir_jun2022 = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/' 
    
    #clean some memory

    del mc
    del mcjpsix
    del data
    del passing
    del failing
    del xx
    del main_df
    del x_train_tmp
    del x_val_tmp
    del x_train
    del x_val
    
    for k in sample_names:        
    #for k in ['data','jpsi_mu']:        
    #for k in ['data']:        
    #for k in ['jpsi_mu']:        
    #for k in ['datalowmass']:        
        print("Loading "+ k)
        samples[k] = read_root(tree_dir_jun2022 + k + '_nopresel_withpresel_v2.root',tree_name)

        samples[k] = to_define(samples[k])
        transf = qt.transform(samples[k][features])
        
        #tmp, qt1 = norm(samples[k])
        #nn = model.predict(tmp)
        #print("SAMPLE AND TRANSFORMATE",samples[k][features],transf)
        #print("MEAN OF THE FEATURES",qt1.center_,qt1.scale_)
        print("MEAN OF THE OLD FEATURES",qt.center_,qt.scale_)

        #tmp, qt = norm(samples[k])
        nn = model.predict(transf)
        print("output",nn)
        if nn[0] == np.nan:
            print("ERROR!")
            break
        # _3 prebug
        # _6 with complete data
        samples[k].loc[:,'nn_onlymc_38'] = nn
        samples[k].loc[:,'fakerate_onlymc_38'] = nn/(1-nn)
        
        #mean = samples[k]['fakerate_data_5'].mean()
        #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_nopresel_withpresel_v2.root', key='BTo3Mu')
        '''
    '''
    # with preselection
    k = 'data'
    #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_prepresel_v2_withnn.root',tree_name)
    samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_prepresel_v2_withnn.root',tree_name)
    samples[k].index = np.array(range(len(samples[k])))
    samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]

    samples[k] = to_define(samples[k])
    transf = qt.transform(samples[k][features])
    print("SAMPLE AND TRANSFORMATE",samples[k][features],transf)
    #tmp, qt = norm(samples[k])
    nn = model.predict(transf)
    # _3 prebug
    # _6 with complete data
    samples[k].loc[:,'nn_data_10'] = nn
    samples[k].loc[:,'fakerate_data_10'] = nn/(1-nn)
    samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'+k+'_withpres_withnn.root', key='BTo3Mu')
    #samples[k][features].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_withpres_withnn_features.root', key='BTo3Mu')
    #transf_df = pd.DataFrame(transf, columns = features)
    #transf_df.to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_withpres_withnn_transf.root', key='BTo3Mu')
        
    print("Uploading data 2")
    # no preselection
    k = 'data'
    #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_ptmax_merged.root',tree_name)
    samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_ptmax_merged_fakerate.root',tree_name)
    samples[k].index = np.array(range(len(samples[k])))
    samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]

    samples[k] = to_define(samples[k])
    transf = qt.transform(samples[k][features])
    print("SAMPLE AND TRANSFORMATE",samples[k][features],transf)
    #tmp, qt = norm(samples[k])
    nn = model.predict(transf)
    # _3 prebug
    # _6 with complete data
    samples[k].loc[:,'nn_data_10'] = nn
    samples[k].loc[:,'fakerate_data_10'] = nn/(1-nn)
    #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_nopres_withnn.root', key='BTo3Mu')
    samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'+k+'_nopres_withnn.root', key='BTo3Mu')
    #samples[k][features].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_nopres_withnn_features.root', key='BTo3Mu')
    #transf_df = pd.DataFrame(transf, columns = features)
    #transf_df.to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_nopres_withnn_transf.root', key='BTo3Mu')

    #mean = samples[k]['fakerate_data_5'].mean()
    #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
    #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_with_mc_corrections.root', key='BTo3Mu')
    #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_withnn.root', key='BTo3Mu')
    '''
    '''
    for k in ['data']:        
        print("Loading "+ k)
        if k == 'data':
            #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_old_trigger_2022Jun02/data_ptmax_merged.root',tree_name)
            #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Oct25/data_ptmax_merged.root',tree_name)
            samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_ptmax_merged.root',tree_name)
            #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_prepresel_v2.root',tree_name)
            samples[k].index = np.array(range(len(samples[k])))
            samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]
        else:
            samples[k] = read_root(tree_dir_dec2021 + k + '_with_mc_corrections.root',tree_name)
            samples[k]['is_mc']= [1 for i in range(len(samples[k].Bmass))]

        samples[k] = to_define(samples[k])
        transf = qt.transform(samples[k][features])
        print("SAMPLE AND TRANSFORMATE",samples[k][features],transf)
        #tmp, qt = norm(samples[k])
        nn = model.predict(transf)
        # _3 prebug
        # _6 with complete data
        samples[k].loc[:,'nn_data_10'] = nn
        samples[k].loc[:,'fakerate_data_10'] = nn/(1-nn)
        
        #mean = samples[k]['fakerate_data_5'].mean()
        #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_with_mc_corrections.root', key='BTo3Mu')
        #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_withnn.root', key='BTo3Mu')

    for k in ['data']:        
        print("Loading "+ k)
        if k == 'data':
            #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_old_trigger_2022Jun02/data_ptmax_merged.root',tree_name)
            #samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Oct25/data_ptmax_merged.root',tree_name)
            samples[k] = read_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_prepresel_v2.root',tree_name)
            samples[k].index = np.array(range(len(samples[k])))
            samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]
        else:
            samples[k] = read_root(tree_dir_dec2021 + k + '_with_mc_corrections.root',tree_name)
            samples[k]['is_mc']= [1 for i in range(len(samples[k].Bmass))]

        
        samples[k] = to_define(samples[k])
        transf = qt.transform(samples[k][features])
        #tmp, qt = norm(samples[k])
        print("SAMPLE AND TRANSFORMATE",samples[k][features],transf)
        nn = model.predict(transf)
        # _3 prebug
        # _6 with complete data
        samples[k].loc[:,'nn_data_10'] = nn
        samples[k].loc[:,'fakerate_data_10'] = nn/(1-nn)
        
        #mean = samples[k]['fakerate_data_5'].mean()
        #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_with_mc_corrections.root', key='BTo3Mu')
        #samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'+k+'_withnn.root', key='BTo3Mu')
    '''
    '''
    for i,k,path in zip([0,1],['data','data'],['/pnfs/psi.ch/cms/trivcat//store/user/friti/dataframes_old_trigger_2022Jun02/data/data_UL_0_ptmax.root','/pnfs/psi.ch/cms/trivcat//store/user/friti/dataframes_2022Jun02/data/data_UL_0_ptmax.root']):        
        print("Loading "+ k)
        if k == 'data':
            samples[k] = read_root(path,tree_name)
            samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]
        else:
            samples[k] = read_root(tree_dir_dec2021 + k + '_with_mc_corrections.root',tree_name)
            samples[k]['is_mc']= [1 for i in range(len(samples[k].Bmass))]

        samples[k] = to_define(samples[k])
        tmp, qt = norm(samples[k])
        nn = model.predict(tmp)
        # _3 prebug
        # _6 with complete data
        samples[k].loc[:,'nn_data_8'] = nn
        samples[k].loc[:,'fakerate_data_8'] = nn/(1-nn)
        
        #mean = samples[k]['fakerate_data_5'].mean()
        #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        samples[k].to_root('data'+str(i)+'_withnn.root', key='BTo3Mu')
    '''
    '''
    k = 'taunu'
    samples[k] = read_root(tree_dir_dec2021 +'/BcToJpsiTauNu_trigger.root',tree_name)
    samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]
    samples[k] = to_define(samples[k])
    tmp, qt = norm(samples[k])
    nn = model.predict(tmp)
    samples[k].loc[:,'nn_data_2'] = nn
    samples[k].loc[:,'fakerate_data_2'] = nn/(1-nn)
    
    mean = samples[k]['fakerate_data_2'].mean()
    samples[k].loc[:,'fakerate_data_mean_2'] = [1./mean for i in range(len(samples[k]['Bmass']))]
    samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'+k+'_fakerate_only_iso.root', key='BTo3Mu')
    
    #samples['data_for_comb'].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_ptmax_merged_fakerate.root', key='BTo3Mu')    
    #samples['data_lowmass_for_comb'].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/datalowmass_ptmax_merged_fakerate.root', key='BTo3Mu')
    '''
