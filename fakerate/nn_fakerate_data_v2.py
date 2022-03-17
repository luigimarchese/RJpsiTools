'''
Script that computes the NN to find dependencies of fr to other variables
Also performs the closure test
- train with data and MC
'''
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
from selections_for_fakerate import prepreselection
from histos_nordf import histos #histos file NO root dataframes
#from samples import sample_names
from samples import sample_names_explicit_jpsimother_compressed as sample_names

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
    return xx,Y,main_df

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

    data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_fakerate_only_iso.root'
    mc_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/jpsi_mu_fakerate_only_iso_prepresel.root'
    
    #output path
    nn_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/nn/'
    
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    final_nn_path = nn_path + label 
    
    features = [
        'kpt',
        'keta',
        'kphi',
        #'kmass',
        'mu1pt',
        'mu1eta',
        'mu1phi',
        #'mu1mass',
        'mu2pt',
        'mu2eta',
        #'mu2mass',
        'mu2phi',
        #'Bmass',
        'jpsiK_mass',
        'Q_sq',
        'is_mc',
    ]
    
    os.system('mkdir -p '+ final_nn_path + '/model/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/norm/')
    
    #preselection and not-true muon request
    data = read_root(data_path, 'BTo3Mu', where=prepreselection)
    data = to_define(data)
    data['is_mc']= [0 for i in range(len(data.Bmass))]

    mc = read_root(mc_path, 'BTo3Mu', where=prepreselection + " & (abs(k_genpdgId)==13)")
    mc = to_define(mc)
    mc['is_mc']= [1 for i in range(len(mc.Bmass))]
    
    print("LEN before ",len(data.Bmass))
    print(len(mc.Bmass))

    data = pd.concat([data,mc], ignore_index=True)
    print(len(data.Bmass))
    pass_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel<0.2' 
    fail_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel>0.2'

    passing_tmp   = data.query(pass_id).copy()
    failing_tmp   = data.query(fail_id).copy()
    print(len(passing_tmp.Bmass),len(failing_tmp.Bmass))
    passing_tmp.loc[:,'target'] = np.ones (passing_tmp.shape[0]).astype(int)
    failing_tmp.loc[:,'target'] = np.zeros (failing_tmp.shape[0]).astype(int)
    
    #reindexing to know which one I used for training, so the rest I use for the closure test
    passing_tmp.index = np.array(range(len(passing_tmp)))
    failing_tmp.index = np.array(range(len(failing_tmp)))
    
    passing = passing_tmp.sample(frac = 0.7, replace=False, random_state=1986)
    failing = failing_tmp.sample(frac = 0.7, replace=False, random_state=1986)
    
    #samples for the closure test
    passing_ct = passing_tmp.drop(passing.index)
    failing_ct = failing_tmp.drop(failing.index)

    print(passing,failing)
    xx, Y, main_df = preprocessing(passing,failing)
        
    print("##########################")
    print("########  Model  #########")
    print("##########################")
    
    activation = 'relu'
    
    # define the net
    input  = Input((len(features),))
    layer  = Dense(64, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
    output = Dense(  1, activation='sigmoid', name='output', )(layer)
    
    # Define outputs of your model
    model = Model(input, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'acc'])
    
    print(model.summary())
    
    # plot the models
    # https://keras.io/visualization/
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/'.join([final_nn_path, '/model/model.png']) )
    
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
    
    #print(len(xx[0]),len(Y))
    x_train, x_test, y_train, y_test = train_test_split(xx, Y, test_size=0.2, shuffle= True)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=50, callbacks=callbacks, batch_size=32, verbose=True)  
    
    
    # calculate predictions on the main_df sample
    print('predicting on', main_df.shape[0], 'events')
    x = pd.DataFrame(main_df, columns=features)
    # y = model.predict(x)
    # load the transformation with the correct parameters!
    qt = pickle.load(open('/'.join([final_nn_path, 'input_tranformation_weighted.pck']), 'rb' ))
    x_train = qt.transform(x[features])
    y_pred = model.predict(x_train)
    
    
    # impose norm conservation if you want probabilities
    # compute the overall rescaling factor scale
    scale = 1.
    # scale = np.sum(passing['target']) / np.sum(y)
    # add the score to the main_df sample
    main_df.insert(len(main_df.columns), 'fr', scale * y_pred)

    ####################################
    ###### ROC CURVE ###################
    ####################################
    
    # let sklearn do the heavy lifting and compute the ROC curves for you
    fpr, tpr, wps = roc_curve(main_df.target, main_df.fr) 
    plt.plot(fpr, tpr)
    xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
    plt.plot(xy, xy, color='grey', linestyle='--')
    plt.yscale('linear')
    plt.savefig('/'.join([final_nn_path, '/model/roc_weighted.pdf']) )
    
    # save model and weights
    model.save('/'.join([final_nn_path, '/model/net_model_weighted.h5']) )
    # model.save_weights('net_model_weights.h5')
    
    # rename branches, if you want
    # main_df.rename(
    #     index=str, 
    #     columns={'cand_refit_mass12': 'mass12',}, 
    #     inplace=True)
    
    # save ntuple
    main_df.to_root('/'.join([final_nn_path, 'output_ntuple_weighted.root']), key='tree', store_index=False)
    
    
    print("################################")
    print("########   Closure Test  #######")
    print("################################")
    ###### Closure test #######
    x_clos, y_clos, main_df_ct = preprocessing(passing_ct,failing_ct)

    main_df_ct.loc[:,'nn'] = model.predict(x_clos)
    
    c1 = ROOT.TCanvas('c1', '', 700, 700)
    c1.Draw()
    c1.cd()
    main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
    main_pad.Draw()
    main_pad.SetTicks(True)
    main_pad.SetBottomMargin(0.2)
    
    closure_test(main_df_ct[main_df_ct.target == 1], main_df_ct[main_df_ct.target == 0])


    ###################################
    ####### TEST overfitting ##########
    ###################################

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    #plot
    h1 = ROOT.TH1F("","",30,0,1)
    h2 = ROOT.TH1F("","",30,0,1)
    for t,b in zip(train_pred,test_pred):
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

    c1.SaveAs(final_nn_path + "/model/KS_overfitting_test.png")

    ks_score = h1.KolmogorovTest(h2)
    #ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
    print("KS score: ",ks_score, len(train_pred),len(test_pred))
    #print("KS fail rw: ",ks_fail_rw)


    print("###########################################")
    print("########  Add NN branch in samples  #######")
    print("###########################################")

    # open all samples as pandas
    samples = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    tree_dir_dec2021 = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/' # same samples as Oct21, but new NN for fakerate

    '''for k in sample_names:        
    #for k in ['datalowmass']:        
        print("Loading "+ k)
        if k == 'data':
            samples[k] = read_root(tree_dir_dec2021 + k + '_fakerate_only_iso.root',tree_name)
            samples[k]['is_mc']= [0 for i in range(len(samples[k].Bmass))]
        else:
            samples[k] = read_root(tree_dir_dec2021 + k + '_fakerate_only_iso.root',tree_name)
            samples[k]['is_mc']= [1 for i in range(len(samples[k].Bmass))]
        samples[k] = to_define(samples[k])
        tmp, qt = norm(samples[k])
        nn = model.predict(tmp)
        samples[k].loc[:,'nn_data_2'] = nn
        samples[k].loc[:,'fakerate_data_2'] = nn/(1-nn)
        
        mean = samples[k]['fakerate_data_2'].mean()
        samples[k].loc[:,'fakerate_data_mean_2'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'+k+'_fakerate_only_iso.root', key='BTo3Mu')
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

