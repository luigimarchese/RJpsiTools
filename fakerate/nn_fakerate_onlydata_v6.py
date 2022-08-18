'''
Script that computes the NN to find dependencies of fr to other variables
Also performs the closure test
- train with data and MC
'''

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

#no pop-up windows
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
officialStyle(ROOT.gStyle, ROOT.TGaxis)

epochs = 50
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
    print(X)
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

    data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_nopresel_withpresel_v2.root'

    #output path
    nn_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/nn/'
    
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    final_nn_path = nn_path + label 
    print("Label: ",label)
    features = [
        #'Bpt_reco_log',
        'Bpt_log',
        'abs_Beta',
        'Q_sq',
        'nPV',
        'jpsivtx_log10_lxy_sig',
        'Bmass',
        #'keta',
        #'mu1eta',
        #'Beta',
        #'Bmass',
        #'nPV',
        #'ptcone_k',
        #'keta',
        #'decay_time_pv_jpsi',
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
        #'jpsivtx_log10_lxy_sig',
        #'DR_mu1mu2',
        #'is_mc',
        #'is_mc_jpsix',
    ]
    
    os.system('mkdir -p '+ final_nn_path + '/model/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/')
    os.system('mkdir -p '+ final_nn_path + '/closure_test/norm/')
    
    print("Hello!")
    
    prepreselection = preprepreselection + "&" +triggerselection +"&"+etaselection + "& Bpt_reco<80"
    #preselection and not-true muon request
    data = read_root(data_path, 'BTo3Mu', where=prepreselection )
    #data.index = np.array(range(len(data)))
    data = to_define(data)
    main_df = data
    '''
    print("data before preprocessing",data)
    xx, qt, Y, main_df = preprocessing_whole_sample(data)
    print("data after preprocessing",main_df)
    print("MEAN OF THE FEATURES",qt.center_,qt.scale_)
    '''
    pass_id = 'k_mediumID<0.5 & k_raw_db_corr_iso03_rel<0.2' 
    fail_id = '(k_mediumID<0.5) & (k_raw_db_corr_iso03_rel>0.2)'

    #main_df is already shuffled
    passing_tmp   = main_df.query(pass_id).copy()
    failing_tmp   = main_df.query(fail_id).copy()

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
    print(len(passing_ct.Q_sq),len(failing_ct.Q_sq))
    print(passing,failing)
    '''
    print("passing before preprocessing",passing_tmp)
    
    xx, Y, sample_weight = preprocessing_just_scale(passing_tmp, failing_tmp, qt)
    print("passing after preprocessing",passing_tmp)
    '''
    xx, qt, Y, main_df = preprocessing(passing,failing)

    print("##########################")
    print("########  Scatter plot  #########")
    print("##########################")

    c1=ROOT.TCanvas()
    h_pre = ROOT.TH2F("","",100,0,3,100,0,11)
    for f0,f1 in zip(main_df[features[0]],main_df[features[1]]):
        h_pre.Fill(f0,f1)
    print(h_pre.Integral())
    c1.Draw()
    h_pre.SetTitle("Scatter Plot 2D Pre Scaler;%s;%s"%(features[0],features[1]))
    h_pre.Draw("coltz")

    c1.SaveAs(final_nn_path+'/model/scatter2d_pre.png')

    h_post = ROOT.TH2F("","",100,-2,2,100,-3,2)
    for item in xx:
        h_post.Fill(item[0],item[1])
    
    c1.Draw()
    h_post.SetTitle("Scatter Plot 2D Post Scaler;%s;%s"%(features[0],features[1]))
    h_post.Draw("coltz")
    print(final_nn_path)
    c1.SaveAs(final_nn_path+ '/model/scatter2d_post.png')

    
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
    
    #x_train, x_test, y_train, y_test = train_test_split(xx, Y, test_size=0.2, shuffle= True)

    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)
    
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

    history = model.fit(x_train, y_train,  validation_data=(x_val, y_val),epochs=epochs, callbacks=callbacks, batch_size=32, verbose=True)  
    
    '''
    # calculate predictions on the main_df sample
    print('predicting on', main_df.shape[0], 'events')
    x = pd.DataFrame(main_df, columns=features)
    # y = model.predict(x)
    # load the transformation with the correct parameters!
    qt = pickle.load(open('/'.join([final_nn_path, 'input_tranformation_weighted.pck']), 'rb' ))
    main_train = qt.transform(x[features])
    y_pred = model.predict(main_train)

    #print("predictions",y_pred)
    #print("MEAN OF THE FEATURES",qt.center_,qt.scale_)
    
    # impose norm conservation if you want probabilities
    # compute the overall rescaling factor scale
    scale = 1.
    # scale = np.sum(passing['target']) / np.sum(y)
    # add the score to the main_df sample
    main_train.insert(len(main_train.columns), 'fr', scale * y_pred)
    '''

    ####################################
    ###### ROC CURVE ###################
    ####################################
    train_pred = model.predict(x_train)
    # let sklearn do the heavy lifting and compute the ROC curves for you
    fpr, tpr, wps = roc_curve(y_train, train_pred) 
    plt.plot(fpr, tpr, label='train ROC')
    print("AUC train",sk.metrics.auc(fpr,tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    test_pred = model.predict(x_test)
    fpr, tpr, wps = roc_curve(y_test, test_pred) 
    plt.plot(fpr, tpr, label='test ROC')
    print("AUC test",sk.metrics.auc(fpr,tpr))

    xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
    plt.plot(xy, xy, color='grey', linestyle='--')
    plt.yscale('linear')
    plt.legend()
    plt.savefig('/'.join([final_nn_path, '/model/roc_weighted.pdf']) )
    print("RoC cuerve saved")

    plt.clf()

    #xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
    #plt.plot(xy, xy, color='grey', linestyle='--')

    # save model and weights
    model.save('/'.join([final_nn_path, '/model/net_model_weighted.h5']) )
    # model.save_weights('net_model_weights.h5')
    
    # rename branches, if you want
    # main_df.rename(
    #     index=str, 
    #     columns={'cand_refit_mass12': 'mass12',}, 
    #     inplace=True)
    
    # save ntuple
    #main_df.to_root('/'.join([final_nn_path, 'output_ntuple_weighted.root']), key='tree', store_index=False)
    
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


    ###################################
    ####### TEST overfitting ##########
    ###################################

    train_pred_pass = model.predict(x_train_pass)
    test_pred_pass = model.predict(x_test_pass)

    #plot
    h1 = ROOT.TH1F("","train",30,-0.5,0.5)
    h2 = ROOT.TH1F("","test",30,-0.5,0.5)
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

    c1.SaveAs(final_nn_path + "/model/KS_overfitting_test_pass.png")

    #ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
    print("KS score: ",ks_score, len(train_pred),len(test_pred))
    #print("KS fail rw: ",ks_fail_rw)

    train_pred_fail = model.predict(x_train_fail)
    test_pred_fail = model.predict(x_test_fail)

    #plot
    h1 = ROOT.TH1F("","train",30,-0.5,0.5)
    h2 = ROOT.TH1F("","test",30,-0.5,0.5)
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

    c1.SaveAs(final_nn_path + "/model/KS_overfitting_test_fail.png")

    #ks_fail_rw = h_ratio_passwpass.KolmogorovTest(h_ratio_passpass)
    print("KS score: ",ks_score, len(train_pred),len(test_pred))
    #print("KS fail rw: ",ks_fail_rw)


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


    '''
    print("###########################################")
    print("########  Add NN branch in samples  #######")
    print("###########################################")

    # open all samples as pandas
    samples = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    #tree_dir_dec2021 = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/' # same samples as Oct21, but new NN for fakerate
    tree_dir_jun2022 = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/' 
    

    for k in sample_names:        
    #for k in ['data','jpsi_mu']:
    #for k in ['datalowmass']:        
        samples[k] = read_root(tree_dir_jun2022 + k + '_nopresel_withpresel_v2.root',tree_name)
        print("Loading "+ k)
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
        samples[k].loc[:,'nn_onlydata_32'] = nn
        samples[k].loc[:,'fakerate_onlydata_32'] = nn/(1-nn)
        
        #mean = samples[k]['fakerate_data_5'].mean()
        #samples[k].loc[:,'fakerate_data_mean_5'] = [1./mean for i in range(len(samples[k]['Bmass']))]
        samples[k].to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'+k+'_nopresel_withpresel_v2.root', key='BTo3Mu')

    '''
