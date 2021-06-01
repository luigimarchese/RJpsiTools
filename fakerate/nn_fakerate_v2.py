'''
Script that computes the NN to find dependencies of fr to other variables
Also performs the closure test

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
from selections import preselection, pass_id, fail_id
from histos_nordf import histos #histos file NO root dataframes

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
    mc_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_trigger_bcclean.root'
    #mc_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_tr_bc_newb.root'
    #output path
    nn_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/nn/'
    
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    final_nn_path = nn_path + label 
    
    features = [
        'kpt',
        'keta',
        'kphi',
        'kmass',
        'mu1pt',
        'mu1eta',
        'mu1phi',
        'mu1mass',
        'mu2pt',
        'mu2eta',
        'mu2mass',
        'mu2phi',
        #'dr12',
        #'dr23',
        #'dr13'
        'Bmass',
        #'Bpt_reco',
    ]
    
    os.system('mkdir -p '+ final_nn_path + '/model/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/norm/')
    
    #preselection and not-true muon request
    mc = read_root(mc_path, 'BTo3Mu', where=preselection + '& !(abs(k_genpdgId)==13)')
    mc = to_define(mc)
    
    passing_mc_tmp   = mc.query(pass_id).copy()
    failing_mc_tmp   = mc.query(fail_id).copy()
    
    passing_mc_tmp.loc[:,'target'] = np.ones (passing_mc_tmp.shape[0]).astype(int)
    failing_mc_tmp.loc[:,'target'] = np.zeros (failing_mc_tmp.shape[0]).astype(int)
    
    #reindexing to know which one I used for training, so the rest I use for the closure test
    passing_mc_tmp.index = np.array(range(len(passing_mc_tmp)))
    failing_mc_tmp.index = np.array(range(len(failing_mc_tmp)))
    
    passing_mc = passing_mc_tmp.sample(frac = 0.7, replace=False, random_state=1986)
    failing_mc = failing_mc_tmp.sample(frac = 0.7, replace=False, random_state=1986)
    
    #samples for the closure test
    passing_mc_ct = passing_mc_tmp.drop(passing_mc.index)
    failing_mc_ct = failing_mc_tmp.drop(failing_mc.index)
    
    xx, Y, main_df = preprocessing(passing_mc,failing_mc)
    
    
    print("##########################")
    print("########  Model  #########")
    print("##########################")
    
    # activation = 'tanh'
    #activation = 'selu'
    # activation = 'sigmoid'
    activation = 'relu'
    # activation = 'LeakyReLU' #??????
    
    # define the net
    input  = Input((len(features),))
    layer  = Dense(16, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
    
    '''layer  = Dense(256, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
    #         layer  = Dropout(0., name='dropout1')(layer)
    layer  = BatchNormalization()(layer)
    layer  = Dense(64, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer)
    layer  = Dropout(0.3, name='dropout2')(layer)
    layer  = BatchNormalization()(layer)
    layer  = Dense(64, activation=activation   , name='dense3', kernel_constraint=unit_norm())(layer)
    layer  = Dropout(0.4, name='dropout3')(layer)
    layer  = BatchNormalization()(layer)
    layer  = Dense(64, activation=activation   , name='dense4', kernel_constraint=unit_norm())(layer)
    layer  = Dropout(0.5, name='dropout4')(layer)
    layer  = BatchNormalization()(layer)
    layer  = Dense(64, activation=activation   , name='dense5', kernel_constraint=unit_norm())(layer)
    layer  = Dropout(0.8, name='dropout5')(layer)
    layer  = BatchNormalization()(layer)
    '''
    output = Dense(  1, activation='sigmoid', name='output', )(layer)
    
    # Define outputs of your model
    model = Model(input, output)
    
    # choose your optimizer
    # opt = SGD(lr=0.0001, momentum=0.8)
    # opt = Adam(lr=0.001, decay=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = Adam(lr=0.01, decay=0.05, beta_1=0.9, beta_2=0.999, amsgrad=True)
    # opt = 'Adam'
    
    # compile and choose your loss function (binary cross entropy for a 1-0 classification problem)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae', 'acc'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'acc'])
    
    # print net summary
    print(model.summary())
    
    # plot the models
    # https://keras.io/visualization/
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='/'.join([final_nn_path, '/model/model.png']) )
    
    # save the exact list of features
    pickle.dump( features, open( '/'.join([final_nn_path, '/model/input_features.pck']), 'wb' ) )
    
    # early stopping
    # monitor = 'val_acc'
    monitor = 'val_loss'
    # monitor = 'val_mae'
    es = EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=50, restore_best_weights=True)
    
    # reduce learning rate when at plateau, fine search the minimum
    reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='auto', factor=0.2, patience=5, min_lr=0.00001, cooldown=10, verbose=True)
    
    # save the model every now and then
    filepath = '/'.join([final_nn_path, '/model/saved-model-{epoch:04d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.h5'])
    save_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    # train only the classifier. beta is set at 0 and the discriminator is not trained
    callbacks = [reduce_lr, save_model]
    x_train, x_val, y_train, y_val = train_test_split(xx, Y, test_size=0.2, shuffle= True)
    
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=20, callbacks=callbacks, batch_size=32, verbose=True)  
    
    
    # plot loss function trends for train and validation sample
    #plt.clf()
    #plt.title('loss')
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    # plt.yscale('log')
    #center = min(history.history['val_loss'] + history.history['loss'])
    #plt.ylim((center*0.98, center*1.5))
    #plt.grid(True)
    #plt.savefig('/'.join([final_nn_path, '/model/loss_function_history_weighted.pdf']))
    #plt.clf()
    
    '''
    # plot accuracy trends for train and validation sample
    plt.title('accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    center = max(history.history['val_acc'] +  history.history['acc'])
    plt.ylim((center*0.85, center*1.02))
    # plt.yscale('log')
    plt.grid(True)
    plt.savefig('/'.join([final_nn_path, '/model/accuracy_history_weighted.pdf']) )
    plt.clf()
    
    # plot accuracy trends for train and validation sample
    plt.title('mean absolute error')
    plt.plot(history.history['mae'], label='train')
    plt.plot(history.history['val_mae'], label='test')
    plt.legend()
    center = min(history.history['val_mae'] + history.history['mae'])
    plt.ylim((center*0.98, center*1.5))
    # plt.yscale('log')
    plt.grid(True)
    plt.savefig('/'.join([final_nn_path, '/model/mean_absolute_error_history_weighted.pdf']) )
    plt.clf()
    
    '''
    
    
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
    
    # let sklearn do the heavy lifting and compute the ROC curves for you
    '''fpr, tpr, wps = roc_curve(main_df.target, main_df.fr) 
    plt.plot(fpr, tpr)
    xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
    plt.plot(xy, xy, color='grey', linestyle='--')
    plt.yscale('linear')
    plt.savefig('/'.join([final_nn_path, '/model/roc_weighted.pdf']) )
    '''
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
    x_test, y_test, main_df_ct = preprocessing(passing_mc_ct,failing_mc_ct)
    main_df_ct.loc[:,'nn'] = model.predict(x_test)
    
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

    # open all samples as pandas
    samples = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    tree_dir_data = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #data
    tree_dir_bc = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Apr14' #Bc
    tree_hbmu = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar16' #Hb mu filter

    samples['jpsi_tau'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_jpsi_tau_merged.root', tree_name) #, where=preselection)
    samples['jpsi_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_jpsi_mu_merged.root', tree_name) #, where=preselection)
    samples['chic0_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_chic0_mu_merged.root', tree_name) #, where=preselection)
    samples['chic1_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_chic1_mu_merged.root', tree_name) #, where=preselection)
    samples['chic2_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_chic2_mu_merged.root', tree_name) #, where=preselection)
    samples['hc_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_hc_mu_merged.root', tree_name) #, where=preselection)
    samples['jpsi_hc'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_jpsi_hc_merged.root', tree_name) #, where=preselection)
    samples['psi2s_mu'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_psi2s_mu_merged.root', tree_name) #, where=preselection)
    samples['psi2s_tau'] = read_root(tree_dir_bc + '/BcToJPsiMuMu_is_psi2s_tau_merged.root', tree_name)
    samples['jpsi_x_mu'] = read_root(tree_hbmu + '/HbToJPsiMuMu3MuFilter_trigger_bcclean.root', tree_name) #, where=preselection)
    samples['data'] = read_root(tree_dir_data + '/data_flagtriggersel.root', tree_name)#,where = preselection)
    # preprocess

    # add NN branch
    for k in samples:
        print(k)
        tmp, qt = norm(samples[k])
        samples[k].loc[:,'nn'] = model.predict(tmp)
        #print(samples[k]['nn'])
    # add fr weight branch
    for k in samples:
        tmp = []
        for nn in samples[k]['nn']:
            tmp.append(nn/(1-nn))    
        samples[k].loc[:,'fakerate_weight'] = tmp
    # save them
    for sample in samples:
        samples[sample].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn/'+sample+'_fakerate.root', key='BTo3Mu')
