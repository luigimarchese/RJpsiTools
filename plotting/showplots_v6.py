'''
This script does the final histograms for the fit for the rjpsi analysis
 - Computes the fakes from the fail region
 - Computes the shape uncertainties
 - Multiplies the vents for all the weights
 - Saves png, pdf and .root files with the histos in pass and fail regions + all the shape nuisances

Difference from _v5:
- new option: add_hm_categories
  - it adds the high mass categories for the final fit
Difference from _v4:
- new option for jpsiXMu bkg to be splitted in the different contributions: 
   - FIXME: the datacard production gives an error: I will solve this when we have the new MC, because now we don't need that function

'''
#system
import os
import copy
from datetime import datetime
import random
import time
import sys

# computation libraries
import ROOT
import pandas as pd
import numpy as np
from array import array
import pickle
import math 
from bokeh.palettes import viridis, all_palettes
from keras.models import load_model

# cms libs
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

# personal libs
from histos import histos as histos_lm
from new_branches import to_define
from samples import weights, titles, colours
from selections import preselection, preselection_mc, pass_id, fail_id
from create_datacard_v2 import create_datacard_ch1, create_datacard_ch2, create_datacard_ch3, create_datacard_ch4, create_datacard_ch1_onlypass, create_datacard_ch3_onlypass
from plot_shape_nuisances import plot_shape_nuisances

shape_nuisances = True
flat_fakerate = False # false mean that we use the NN weights for the fr
compute_sf = False # compute scale factors SHAPE nuisances
compute_sf_onlynorm = False # compute only the sf normalisation (best case)
blind_analysis = True
rjpsi = 1

asimov = False
only_pass = False

if asimov:
    blind_analysis=False
    rjpsi = 1

jpsiXmu_bk_explicit = False #true if you want to split the jpsimu bkg contributions depending on the jpsi mother
add_hm_categories = True #true if you want to add also the high mass categories to normalise the jpsimu bkg

if jpsiXmu_bk_explicit:
    from samples import  sample_names_explicit_jpsiXmu as sample_names
else:
    from samples import  sample_names

if add_hm_categories:
    from selections import preselection_hm, preselection_hm_mc
    from histos import histos_hm

start_time = time.time()

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

def make_directories(label):

    if not add_hm_categories:
        channels = ['ch1','ch2']
    else:
        channels = ['ch1','ch2','ch3','ch4']
    
    for ch in channels:
        os.system('mkdir -p plots_ul/%s/%s/pdf/lin/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/pdf/log/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/png/lin/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/png/log/' %(label,ch))
            
    os.system('mkdir -p plots_ul/%s/datacards/' %label)

def save_yields(label, temp_hists):
    with open('plots_ul/%s/yields.txt' %label, 'w') as ff:
        total_expected = 0.
        for kk, vv in temp_hists['norm'].items(): 
            if 'data' not in kk:
                total_expected += vv.Integral()
            print(kk.replace(k, '')[1:], '\t\t%.1f' %vv.Integral(), file=ff)
        print('total expected', '\t%.1f' %total_expected, file=ff)

def save_weights(label, sample_names, weights):
    with open('plots_ul/%s/normalisations.txt' %label, 'w') as ff:
        for sname in sample_names: 
            print(sname+'\t\t%.2f' %weights[sname], file=ff)
        print("Flat fake rate weight %s" %str(flat_fakerate), file = ff)

def save_selection(label, preselection):
    with open('plots_ul/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '): 
            print("    '%s'," %isel, file=ff)
        print('])', file=ff)
        print('pass: %s'%pass_id, file=ff)
        print('fail: %s'%fail_id, file=ff)

def create_legend(temp_hists, sample_names, titles):
    # Legend gymnastics
    leg = ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.SetNColumns(3)
    k = list(temp_hists.keys())[0]
    for kk in sample_names:
        leg.AddEntry(temp_hists[k]['%s_%s' %(k, kk)].GetValue(), titles[kk], 'F' if kk!='data' else 'EP')
    return leg

def create_datacard_prep(hists, shape_hists, shapes_names, sample_names, channel, name, label):
    '''
    Creates and saves the root file with the histograms of each contribution.
    Saves the histograms of the shape nuisances.
    Calls the 'create datacard' function, both for the pass and fail regions, 
    to write the text datacard for the fit in combine. 
    '''
    if only_pass and (channel == 'ch2' or channel == 'ch4'): #don't save the fail datacards
        return

    fout = ROOT.TFile.Open('plots_ul/%s/datacards/datacard_%s_%s.root' %(label, channel, name), 'UPDATE')
    fout.cd()
    myhists = dict()

    for k, v in hists.items():
        for isample in sample_names + ['fakes']:
            if isample in k:
                hh = v.Clone()
                if isample == 'data':
                    hh.SetName(isample+'_obs_'+channel)
                else:
                    hh.SetName(isample+'_'+channel)
                hh.Write()
                myhists[isample] = hh.Clone()
        
    # Creates the shape nuisances both for Pass and Fail regions
    for k,v in shape_hists.items():
        for sname in shapes_names:
            if sname in k:
                hh = v.Clone()
                hh.SetName(sname + '_'+channel)
                hh.Write()

    if only_pass: #the rate of fakes must be == integral in case of only pass category fit, while ==1 in case of two regions
        if channel == 'ch1' :
            create_datacard_ch1_onlypass(myhists,name, label)
        else:
            create_datacard_ch3_onlypass(myhists,name, label)

    else:
        if channel == 'ch1' :
            create_datacard_ch1(myhists,name, label)
        elif channel == 'ch2' :
            create_datacard_ch2(myhists,name, label)
        elif channel == 'ch3' :
            create_datacard_ch3(myhists,name, label)
        else:
            create_datacard_ch4(myhists,name, label)
    fout.Close()

def make_binbybin(hist, sample, channel, label, name):
    if only_pass and (channel == 'ch2' or channel == 'ch4'):
        return

    fout = ROOT.TFile.Open('plots_ul/%s/datacards/datacard_%s_%s.root' %(label, channel, name), 'UPDATE')
    for i in range(1,hist.GetValue().GetNbinsX()+1):
        #histo_up = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+flag+'Up','jpsi_x_mu_bbb'+str(i)+flag+'Up',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        #histo_down = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+flag+'Down','jpsi_x_mu_bbb'+str(i)+flag+'Down',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        histo_up = ROOT.TH1D(sample+'_bbb'+str(i)+channel+'Up_'+channel,'',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        histo_down = ROOT.TH1D(sample+'_bbb'+str(i)+channel+'Down_'+channel,'',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        for nbin in range(1,hist.GetValue().GetNbinsX()+1):
            if nbin == i:
                histo_up.SetBinContent(nbin,hist.GetValue().GetBinContent(nbin) + hist.GetValue().GetBinError(nbin))
                histo_up.SetBinError(nbin,hist.GetValue().GetBinError(nbin) + math.sqrt(hist.GetValue().GetBinError(nbin)))
                histo_down.SetBinContent(nbin,hist.GetValue().GetBinContent(nbin) - hist.GetValue().GetBinError(nbin))
                histo_down.SetBinError(nbin,hist.GetValue().GetBinError(nbin) - math.sqrt(hist.GetValue().GetBinError(nbin)))
            else:
                histo_up.SetBinContent(nbin,hist.GetValue().GetBinContent(nbin))
                histo_up.SetBinError(nbin,hist.GetValue().GetBinError(nbin))
                histo_down.SetBinContent(nbin,hist.GetValue().GetBinContent(nbin))
                histo_down.SetBinError(nbin,hist.GetValue().GetBinError(nbin))
        fout.cd()
        histo_up.Write()
        histo_down.Write()
    fout.Close()

def define_shape_nuisances(sname, shapes, samples, nuisance_name, central_value, up_value, down_value, central_weights_string):
    shapes[sname + '_' + nuisance_name + 'Up'] = samples[sname]
    if sname == 'jpsi_mu':
        shapes[sname + '_' + nuisance_name + 'Up'] = shapes[sname + '_' + nuisance_name + 'Up'].Define('shape_weight_tmp', central_weights_string.replace(central_value,up_value)+'*hammer_bglvar')
    elif sname == 'jpsi_tau':
        shapes[sname + '_' + nuisance_name + 'Up'] = shapes[sname + '_' + nuisance_name + 'Up'].Define('shape_weight_tmp', central_weights_string.replace(central_value,up_value)+'*hammer_bglvar*%f*%f' %(blind,rjpsi))
    elif  'jpsi_x_mu' in sname: #this works both for jpsi_x_mu and for its subsamples
        shapes[sname + '_' + nuisance_name + 'Up'] = shapes[sname + '_' + nuisance_name + 'Up'].Define('shape_weight_tmp', central_weights_string.replace(central_value,up_value)+'*jpsimother_weight')

    else:
        shapes[sname + '_' + nuisance_name + 'Up'] = shapes[sname + '_' + nuisance_name + 'Up'].Define('shape_weight_tmp', central_weights_string.replace(central_value,up_value))

    shapes[sname + '_' + nuisance_name + 'Down'] = samples[sname]
    if sname == 'jpsi_mu':
        shapes[sname + '_' + nuisance_name + 'Down'] = shapes[sname + '_' + nuisance_name + 'Down'].Define('shape_weight_tmp', central_weights_string.replace(central_value,down_value)+'*hammer_bglvar')
    elif sname == 'jpsi_tau':
        shapes[sname + '_' + nuisance_name + 'Down'] = shapes[sname + '_' + nuisance_name + 'Down'].Define('shape_weight_tmp', central_weights_string.replace(central_value,down_value)+'*hammer_bglvar*%f*%f' %(blind,rjpsi))
    elif 'jpsi_x_mu' in sname:
        shapes[sname + '_' + nuisance_name + 'Down'] = shapes[sname + '_' + nuisance_name + 'Down'].Define('shape_weight_tmp', central_weights_string.replace(central_value,down_value)+'*jpsimother_weight')

    else:
        shapes[sname + '_' + nuisance_name + 'Down'] = shapes[sname + '_' + nuisance_name + 'Down'].Define('shape_weight_tmp', central_weights_string.replace(central_value,down_value))
    return shapes[sname + '_' + nuisance_name + 'Up'], shapes[sname + '_' + nuisance_name + 'Down']

# Canvas and Pad gymnastics
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0.25, 1. , 1.  )
main_pad.Draw()
c1.cd()
ratio_pad = ROOT.TPad('ratio_pad', '', 0., 0., 1., 0.25)
ratio_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.)
# main_pad.SetTopMargin(0.3)   
# main_pad.SetLeftMargin(0.15)
# main_pad.SetRightMargin(0.15)
# ratio_pad.SetLeftMargin(0.15)
# ratio_pad.SetRightMargin(0.15)
ratio_pad.SetTopMargin(0.)   
ratio_pad.SetGridy()
ratio_pad.SetBottomMargin(0.45)

##########################################################################################
##########################################################################################

if __name__ == '__main__':
    
    #datacards = ['mu1pt', 'Q_sq', 'm_miss_sq', 'E_mu_star', 'E_mu_canc', 'bdt_tau', 'Bmass', 'mcorr', 'decay_time_ps','k_raw_db_corr_iso04_rel']
    datacards = ['Q_sq']

    # timestamp
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

    # create plot directories
    make_directories(label)
    
    central_weights_string = 'ctau_weight_central*br_weight*puWeight*sf_reco_total*sf_id_jpsi'

    # access the samples, via RDataFrames
    samples_orig = dict()
    samples_lm = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    #tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Oct2021'
    #    tree_dir_hb = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Oct2021'

    '''tree_dir_data = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #data
    tree_dir_hb = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #Hb
    tree_dir_bc = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Apr14' #Bc 
    tree_dir_psitau = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar25' #psi2stau
    tree_hbmu = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar16' #Hb mu filter
    '''
    print("=============================")
    print("====== Loading Samples ======")
    print("=============================")

    for k in sample_names:
        if k == 'data':
            samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_fakerate.root'%(tree_dir,k)) 
        elif k == 'jpsi_mu':
            samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_sf_prepreselection.root'%(tree_dir,k)) 
        #elif k == 'jpsi_x_mu':
        #    samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_sf.root'%(tree_dir_hb,k)) 
        else:
            samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_sf.root'%(tree_dir,k))
        
        #samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_bdtenriched_10.root'%(tree_dir,k))
        #samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_bdt_comb_08.root'%(tree_dir,k))
        #samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_bdt_fakes_01.root'%(tree_dir,k))

    #for k in sample_names:
    #    samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_bdtenriched.root'%(tree_dir,k)) 
    

    #Blind analysis: hide the value of rjpsi for the fit
    if blind_analysis:
        random.seed(2)
        rand = random.randint(0, 10000)
        blind = rand/10000 *1.5 +0.5
    else:
        blind = 1.
    #rjpsi value in case we need to change it 
    
    #################################################
    ####### Weights ################################
    #################################################
    for k, v in samples_orig.items():
        samples_orig[k] = samples_orig[k].Define('br_weight', '%f' %weights[k])
        if k=='jpsi_tau':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar*%f*%f' %(blind,rjpsi))
        elif k=='jpsi_mu':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar')
        elif 'jpsi_x_mu' in k:
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*jpsimother_weight')
        else:
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string  if k!='data' else 'br_weight') 

        #define new columns
        for new_column, new_definition in to_define: 
            if samples_orig[k].HasColumn(new_column):
                continue
            samples_orig[k] = samples_orig[k].Define(new_column, new_definition)

    if flat_fakerate == False:
        for sample in samples_orig:
            samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*nn/(1-nn)') 

    # the scale factor on the id on the third muon only for the PASS region
    for sample in samples_orig:
        samples_orig[sample] = samples_orig[sample].Define('total_weight', 'tmp_weight*sf_id_k' if sample!='data' else 'tmp_weight')
            
    ##############################################
    ##### Preselection ###########################
    ##############################################
    print("===================================")
    print("====== Applying Preselection ======")
    print("===================================")

    #Apply preselection for ch1 and ch2
    for k, v in samples_orig.items():
        filter = preselection_mc if k!='data' else preselection
        samples_lm[k] = samples_orig[k].Filter(filter)
        print(k +" "+str(len(samples_lm[k].AsNumpy()['nPV'])))

    samples_dictionaries = [samples_lm]
    histos_dictionaries = [histos_lm]
   #Apply preselection for ch3 and ch4 (high mass region)
    if add_hm_categories:
        samples_hm = dict()
        print("############################")
        for k, v in samples_orig.items():
            if not (k=='data' or k =='jpsi_x_mu'): #only those samples are different from zero in the high mass region
                continue
            filter = preselection_hm_mc if k!='data' else preselection_hm
            samples_hm[k] = samples_orig[k].Filter(filter)
            print(k +" "+str(len(samples_hm[k].AsNumpy()['nPV'])))

        samples_dictionaries = [samples_lm, samples_hm]
        histos_dictionaries = [histos_lm, histos_hm]

    ###########################################################
    ######### NUISANCES Defininition ##########################
    ###########################################################

    # Shape nuisances definition
    # Create a new disctionary "shapes", similar to the "samples" one defined for the datasets
    # Each entry of the dic is a nuisance for a different dataset
    if shape_nuisances :
        shapes_lm = dict()
        shapes_dictionaries = [shapes_lm]
        if add_hm_categories:
            shapes_hm = dict()
            shapes_dictionaries = [shapes_lm, shapes_hm]

        for iter,(shapes,samples) in enumerate(zip(shapes_dictionaries, samples_dictionaries)):
            ############################
            ########  CTAU  ############
            ############################
        
            for sname in samples:
                if (sname != 'jpsi_x_mu' and sname != 'data' ):    #Only Bc samples want this nuisance
                    shapes[sname + '_ctauUp'], shapes[sname + '_ctauDown'] = define_shape_nuisances(sname, shapes, samples, 'ctau', 'ctau_weight_central', 'ctau_weight_up', 'ctau_weight_down', central_weights_string)
                ###############################
                ########  PILE UP  ############
                ###############################

                if (sname != 'data'): # all MC samples
                    shapes[sname + '_puWeightUp'], shapes[sname + '_puWeightDown'] = define_shape_nuisances(sname,shapes, samples, 'puWeight', 'puWeight', 'puWeightUp', 'puWeightDown', central_weights_string)

                if compute_sf:
                    ###############################
                    ########  SF RECO  ############
                    ###############################
                    if (sname != 'data'):

                        for ireco in range(0, 16):
                            shapes[sname + '_sfReco_'+str(ireco)+'Up'], shapes[sname + '_sfReco_'+str(ireco)+'Down'] = define_shape_nuisances(sname,shapes, samples,  'sfReco_'+str(ireco), 'sf_reco_total', 'sf_reco_'+str(ireco)+'_up', 'sf_reco_'+str(ireco)+'_down', central_weights_string)

                    ###############################
                    ########  SF ID  ##############
                    ###############################
		
                    # Only jpsi for now, bc the sf_id for the third muon is only in the pass region!
                    if (sname != 'data'):
                        for iid in range(0, 16):
                            shapes[sname + '_sfId_'+str(iid)+'Up'], shapes[sname + '_sfId_'+str(iid)+'Down'] = define_shape_nuisances(sname, shapes, samples, 'sfId_'+str(iid), 'sf_id_jpsi', 'sf_id_'+str(iid)+'_jpsi_up', 'sf_id_'+str(iid)+'_jpsi_down', central_weights_string)

                if compute_sf_onlynorm:
                    ###############################
                    ########  SF RECO  ############
                    ###############################
                    if (sname != 'data'):

                        shapes[sname + '_sfReco_Up'], shapes[sname + '_sfReco_Down'] = define_shape_nuisances(sname,shapes, samples,  'sfReco', 'sf_reco_total', 'sf_reco_all_up', 'sf_reco_all_down', central_weights_string)

                    ###############################
                    ########  SF ID  ##############
                    ###############################
		
                    # Only jpsi for now, bc the sf_id for the third muon is only in the pass region!
                    if (sname != 'data'):
                        shapes[sname + '_sfId_Up'], shapes[sname + '_sfId_Down'] = define_shape_nuisances(sname, shapes, samples, 'sfId', 'sf_id_jpsi', 'sf_id_all_jpsi_up', 'sf_id_all_jpsi_down', central_weights_string)

            ######################################
            ########  FORM FACTORS  ##############
            #####################################
            
            if not iter: #iter == 0
                # form factor shape nuisances for jpsi mu and jpsi tau datasets
                hammer_branches = ['hammer_bglvar_e0up',
                                   'hammer_bglvar_e0down',
                                   'hammer_bglvar_e1up',
                                   'hammer_bglvar_e1down',
                                   'hammer_bglvar_e2up',
                                   'hammer_bglvar_e2down',
                                   'hammer_bglvar_e3up',
                                   'hammer_bglvar_e3down',
                                   'hammer_bglvar_e4up',
                                   'hammer_bglvar_e4down',
                                   'hammer_bglvar_e5up',
                                   'hammer_bglvar_e5down',
                                   'hammer_bglvar_e6up',
                                   'hammer_bglvar_e6down',
                                   'hammer_bglvar_e7up',
                                   'hammer_bglvar_e7down',
                                   'hammer_bglvar_e8up',
                                   'hammer_bglvar_e8down',
                                   'hammer_bglvar_e9up',
                                   'hammer_bglvar_e9down',
                                   'hammer_bglvar_e10up',
                                   'hammer_bglvar_e10down'
                               ]

                for ham in hammer_branches:
                    new_name = ham.replace('hammer_','')
                    # Redefinition of the name for combine requests
                    if 'up' in ham:
                        new_name = new_name.replace('up','Up')
                    elif 'down' in ham:
                        new_name = new_name.replace('down','Down')
            
                    shapes['jpsi_mu_'+new_name] = samples['jpsi_mu']
                    shapes['jpsi_mu_'+new_name] = shapes['jpsi_mu_'+new_name].Define('shape_weight_tmp', central_weights_string+'*'+ham)
                    shapes['jpsi_tau_'+new_name] = samples['jpsi_tau']
                    shapes['jpsi_tau_'+new_name] = shapes['jpsi_tau_'+new_name].Define('shape_weight_tmp', central_weights_string+'*'+ham+'*%f*%f' %(blind,rjpsi))
                    

            if flat_fakerate == False:
                for name in shapes:
                    shapes[name] = shapes[name].Define('shape_weight_wfr','shape_weight_tmp*nn/(1-nn)')
        
            # For the Pass region we add the sf_id_k and its shape uncetrtainty
            for name in shapes:
                if 'sfId' not in name:
                    shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_k')
                elif (compute_sf):
                    if 'Up' in name:
                        number = name.split('_')[-1].strip('Up')
                        shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_'+number+'_k_up')
                    elif 'Down' in name:
                        number = name.split('_')[-1].strip('Down')
                        shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_'+number+'_k_down')
                elif (compute_sf_onlynorm):
                    if 'Up' in name:
                        shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_all_k_up')
                    elif 'Down' in name:
                        shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_all_k_down')


    ##################################
    ###### HISTOS ###################
    ##################################

    for iteration,(shapes,samples,histos) in enumerate(zip(shapes_dictionaries, samples_dictionaries, histos_dictionaries)):
        
        if not iteration: #iteration==0
            channels = ['ch1','ch2']
        else:
            channels = ['ch3','ch4']


        # first create all the pointers
        print('====> creating pointers to histo')
        temp_hists      = {} # pass muon ID category
        temp_hists_fake = {} # fail muon ID category
    
        for k, v in histos.items():    
            temp_hists     [k] = {}
            temp_hists_fake[k] = {}
            for kk, vv in samples.items():
                temp_hists     [k]['%s_%s' %(k, kk)] = vv.Filter(pass_id).Histo1D(v[0], k, 'total_weight')
                if flat_fakerate:
                    temp_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight')
                else:
                    temp_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr')
    
        # Create pointers for the shapes histos 
        if shape_nuisances:
            print('====> shape uncertainties histos')
            unc_hists      = {} # pass muon ID category
            unc_hists_fake = {} # pass muon ID category
            for k, v in histos.items():    
                # Compute them only for the variables that we want to fit
                if (k not in datacards and iteration == 0) or (k!='Bmass' and iteration):
                    continue
                unc_hists     [k] = {}
                unc_hists_fake[k] = {}
                for kk, vv in shapes.items():
                    unc_hists     [k]['%s_%s' %(k, kk)] = vv.Filter(pass_id).Histo1D(v[0], k, 'shape_weight')
                    if flat_fakerate:
                        unc_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight')
                    else:
                        unc_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight_wfr')

        print('====> now looping')
        for k, v in histos.items():
            # add bin by bin unc plots for jpsi+mu
            # req jpsi+x+mu
            # loop on the bins and multiply each of them for sqrtN
            # save in the same root file as the other shape nuisances
            for sample,sample_item in samples.items():
                if "jpsi_x_mu" in sample:
                    make_binbybin(temp_hists[k]['%s_%s'%(k,sample)],sample,channels[0], label, k)
                    make_binbybin(temp_hists_fake[k]['%s_%s'%(k,sample)],sample,channels[1], label, k)
        
            #check that bins are not zero (if they are, correct)
            for i, kv in enumerate(temp_hists[k].items()):
                key = kv[0]
                ihist = kv[1]
                sample_name = key.split(k+'_')[1]
                for i in range(1,ihist.GetNbinsX()+1):
                    if ihist.GetBinContent(i) <= 0:
                        ihist.SetBinContent(i,0.0001)

            for i, kv in enumerate(temp_hists_fake[k].items()):
                key = kv[0]
                ihist = kv[1]
                sample_name = key.split(k+'_')[1]
                for i in range(1,ihist.GetNbinsX()+1):
                    if ihist.GetBinContent(i) <= 0:
                        ihist.SetBinContent(i,0.0001)

            if shape_nuisances and ((k in datacards and  iteration==0) or (k == 'Bmass' and iteration)):
                
                for i, kv in enumerate(unc_hists[k].items()):
                    key = kv[0]
                    ihist = kv[1]
                    sample_name = key.split(k+'_')[1]
                    for i in range(1,ihist.GetNbinsX()+1):
                        if ihist.GetBinContent(i) <= 0:
                            ihist.SetBinContent(i,0.0001)

                    for i, kv in enumerate(unc_hists_fake[k].items()):
                        key = kv[0]
                        ihist = kv[1]
                        sample_name = key.split(k+'_')[1]
                    for i in range(1,ihist.GetNbinsX()+1):
                        if ihist.GetBinContent(i) <= 0:
                            ihist.SetBinContent(i,0.0001)

            c1.cd()
            leg = create_legend(temp_hists, [str(k) for k,v in samples.items()], titles)
            main_pad.cd()
            main_pad.SetLogy(False)
        
            maxima = [] # save maxima for the look of the stack plot
            data_max = 0.
            for i, kv in enumerate(temp_hists[k].items()):
                key = kv[0]
                ihist = kv[1]
                sample_name = key.split(k+'_')[1]
                ihist.GetXaxis().SetTitle(v[1])
                ihist.GetYaxis().SetTitle('events')
                
                ihist.SetLineColor(colours[sample_name])
                ihist.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
                if key!='%s_data'%k:
                    maxima.append(ihist.GetMaximum())
                else:
                    data_max = ihist.GetMaximum()
    
            # Definition of stack histos
            ths1      = ROOT.THStack('stack', '')
            ths1_fake = ROOT.THStack('stack_fake', '')

            for i, kv in enumerate(temp_hists[k].items()):
                key = kv[0]
                if key=='%s_data'%k: continue
                ihist = kv[1]
                ihist.SetMaximum(2.*max(maxima))
                ihist.Draw('hist' + 'same'*(i>0))
                ths1.Add(ihist.GetValue())
            
            # apply same aestethics to pass and fail
            for kk in temp_hists[k].keys():
                temp_hists_fake[k][kk].GetXaxis().SetTitle(temp_hists[k][kk].GetXaxis().GetTitle())
                temp_hists_fake[k][kk].GetYaxis().SetTitle(temp_hists[k][kk].GetYaxis().GetTitle())
                temp_hists_fake[k][kk].SetLineColor(temp_hists[k][kk].GetLineColor())
                temp_hists_fake[k][kk].SetFillColor(temp_hists[k][kk].GetFillColor())
                

            # fakes for the fail contribution
            # subtract data to MC
            temp_hists[k]['%s_fakes' %k] = temp_hists_fake[k]['%s_data' %k].Clone()
            fakes = temp_hists[k]['%s_fakes' %k]
            # Subtract to fakes all the contributions of other samples in the fail region
            for i, kv in enumerate(temp_hists_fake[k].items()):
                if 'data' in kv[0]:
                    kv[1].SetLineColor(ROOT.kBlack)
                    continue
                else:
                    fakes.Add(kv[1].GetPtr(), -1.)
            #check fakes do not have <= 0 bins
            for b in range(1,fakes.GetNbinsX()+1):
                if fakes.GetBinContent(b)<=0.:
                    fakes.SetBinContent(b,0.0001)

            fakes.SetFillColor(colours['fakes'])
            fakes.SetFillStyle(1001)
            fakes.SetLineColor(colours['fakes'])
            fakes_forfail = fakes.Clone()
            if flat_fakerate:
                fakes.Scale(weights['fakes'])
            ths1.Add(fakes)
            ths1.Draw('hist')
            try:
                ths1.GetXaxis().SetTitle(v[1])
            except:
                continue
            ths1.GetYaxis().SetTitle('events')
            ths1.SetMaximum(1.6*max(sum(maxima), data_max))
            ths1.SetMinimum(0.)
        
            # statistical uncertainty
            stats = ths1.GetStack().Last().Clone()
            stats.SetLineColor(0)
            stats.SetFillColor(ROOT.kGray+1)
            stats.SetFillStyle(3344)
            stats.SetMarkerSize(0)
            stats.Draw('E2 SAME')
            
            if flat_fakerate:
                leg.AddEntry(fakes, 'fakes flat', 'F')    
            else:
                leg.AddEntry(fakes, 'fakes nn', 'F')    
            leg.AddEntry(stats, 'stat. unc.', 'F')
            leg.Draw('same')
    
            #temp_hists[k]['%s_data'%k].GetXaxis().SetRange(0,14)
            if not asimov:
                temp_hists[k]['%s_data'%k].Draw('EP SAME')
            CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Work in Progress', lumi_13TeV = 'L = 59.7 fb^{-1}')
            main_pad.cd()
            # if the analisis if blind, we don't want to show the rjpsi prefit value
            if not blind_analysis:
                rjpsi_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
                rjpsi_value.AddText('R(J/#Psi) = %.2f' %rjpsi)
                rjpsi_value.SetFillColor(0)
                rjpsi_value.Draw('EP')
        
            # Ratio for pass region
            ratio_pad.cd()
            ratio = temp_hists[k]['%s_data'%k].Clone()
            ratio.SetName(ratio.GetName()+'_ratio')
            ratio.Divide(stats)
            ratio_stats = stats.Clone()
            ratio_stats.SetName(ratio.GetName()+'_ratiostats')
            ratio_stats.Divide(stats)
            ratio_stats.SetMaximum(1.999) # avoid displaying 2, that overlaps with 0 in the main_pad
            ratio_stats.SetMinimum(0.001) # and this is for symmetry
            ratio_stats.GetYaxis().SetTitle('obs/exp')
            ratio_stats.GetYaxis().SetTitleOffset(0.5)
            ratio_stats.GetYaxis().SetNdivisions(405)
            ratio_stats.GetXaxis().SetLabelSize(3.* ratio.GetXaxis().GetLabelSize())
            ratio_stats.GetYaxis().SetLabelSize(3.* ratio.GetYaxis().GetLabelSize())
            ratio_stats.GetXaxis().SetTitleSize(3.* ratio.GetXaxis().GetTitleSize())
            ratio_stats.GetYaxis().SetTitleSize(3.* ratio.GetYaxis().GetTitleSize())
            
            norm_stack = ROOT.THStack('norm_stack', '')
            
            for kk, vv in temp_hists[k].items():
                if 'data' in kk: continue
                hh = vv.Clone()
                hh.Divide(stats)
                norm_stack.Add(hh)
            norm_stack.Draw('hist same')

            line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(1)
            ratio_stats.Draw('E2')
            norm_stack.Draw('hist same')
            ratio_stats.Draw('E2 same')
            line.Draw('same')
            if not asimov:
                ratio.Draw('EP same')
    
            c1.Modified()
            c1.Update()

            c1.SaveAs('plots_ul/%s/%s/pdf/lin/%s.pdf' %(label, channels[0], k))
            c1.SaveAs('plots_ul/%s/%s/png/lin/%s.png' %(label, channels[0], k))
                    
            ths1.SetMaximum(20*max(sum(maxima), data_max))
            ths1.SetMinimum(10)
            main_pad.SetLogy(True)
            c1.Modified()
            c1.Update()

            c1.SaveAs('plots_ul/%s/%s/pdf/log/%s.pdf' %(label, channels[0], k))
            c1.SaveAs('plots_ul/%s/%s/png/log/%s.png' %(label, channels[0], k))
        
            if shape_nuisances and ((k in datacards and  iteration==0) or (k == 'Bmass' and iteration)):
                create_datacard_prep(temp_hists[k], unc_hists[k], shapes, [name for name,v in samples.items()], channels[0], k, label)
                #plot_shape_nuisances(label, k, 'pass', compute_sf = compute_sf, compute_sf_onlynorm = compute_sf_onlynorm)
            #####################################################
            # Now creating and saving the stack of the fail region

            c1.cd()
            main_pad.cd()
            main_pad.SetLogy(False)

            for i, kv in enumerate(temp_hists_fake[k].items()):
                key = kv[0]
                if key=='%s_data'%k: 
                    max_fake = kv[1].GetMaximum()
                    continue
                ihist = kv[1]
                ths1_fake.Add(ihist.GetValue())

            temp_hists_fake[k]['%s_fakes' %k] = fakes_forfail
            ths1_fake.Add(fakes_forfail)
            ths1_fake.Draw('hist')
            ths1_fake.SetMaximum(1.6*max_fake)
            ths1_fake.GetYaxis().SetTitle('events')
            
            stats_fake = ths1_fake.GetStack().Last().Clone()
            stats_fake.SetLineColor(0)
            stats_fake.SetFillColor(ROOT.kGray+1)
            stats_fake.SetFillStyle(3344)
            stats_fake.SetMarkerSize(0)
            stats_fake.Draw('E2 SAME')
            
            if not asimov:
                temp_hists_fake[k]['%s_data'%k].Draw('EP SAME')
            CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
            leg.Draw('same')

            # Ratio for pass region
            
            ratio_pad.cd()
            ratio_fake = temp_hists_fake[k]['%s_data'%k].Clone()
            ratio_fake.SetName(ratio_fake.GetName()+'_ratio')
            ratio_fake.Divide(stats_fake)
            ratio_stats_fake = stats_fake.Clone()
            ratio_stats_fake.SetName(ratio.GetName()+'_ratiostats_fake')
            ratio_stats_fake.Divide(stats_fake)
            ratio_stats_fake.SetMaximum(1.999) # avoid displaying 2, that overlaps with 0 in the main_pad
            ratio_stats_fake.SetMinimum(0.001) # and this is for symmetry
            ratio_stats_fake.GetYaxis().SetTitle('obs/exp')
            ratio_stats_fake.GetYaxis().SetTitleOffset(0.5)
            ratio_stats_fake.GetYaxis().SetNdivisions(405)
            ratio_stats_fake.GetXaxis().SetLabelSize(3.* ratio_fake.GetXaxis().GetLabelSize())
            ratio_stats_fake.GetYaxis().SetLabelSize(3.* ratio_fake.GetYaxis().GetLabelSize())
            ratio_stats_fake.GetXaxis().SetTitleSize(3.* ratio_fake.GetXaxis().GetTitleSize())
            ratio_stats_fake.GetYaxis().SetTitleSize(3.* ratio_fake.GetYaxis().GetTitleSize())
            
            norm_stack_fake = ROOT.THStack('norm_stack', '')

            for kk, vv in temp_hists_fake[k].items():
                if 'data' in kk: continue
                hh = vv.Clone()
                hh.Divide(stats_fake)
                norm_stack_fake.Add(hh)
            norm_stack_fake.Draw('hist same')

            line = ROOT.TLine(ratio_fake.GetXaxis().GetXmin(), 1., ratio_fake.GetXaxis().GetXmax(), 1.)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(1)
            ratio_stats_fake.Draw('E2')
            norm_stack_fake.Draw('hist same')
            ratio_stats_fake.Draw('E2 same')
            line.Draw('same')
            if not asimov:
                ratio_fake.Draw('EP same')

            c1.Modified()
            c1.Update()

            c1.SaveAs('plots_ul/%s/%s/pdf/lin/%s.pdf' %(label, channels[1], k))
            c1.SaveAs('plots_ul/%s/%s/png/lin/%s.png' %(label, channels[1], k))

            ths1_fake.SetMaximum(20*max(sum(maxima), data_max))
            ths1_fake.SetMinimum(10)
            main_pad.SetLogy(True)
            c1.Modified()
            c1.Update()

            c1.SaveAs('plots_ul/%s/%s/pdf/log/%s.pdf' %(label, channels[1], k))
            c1.SaveAs('plots_ul/%s/%s/png/log/%s.png' %(label, channels[1], k))

            if shape_nuisances and ((k in datacards and  iteration==0) or (k == 'Bmass' and iteration)):
                create_datacard_prep(temp_hists_fake[k], unc_hists_fake[k], shapes, [name for name,v in samples.items()], channels[1], k, label)
                #create_datacard_prep(temp_hists_fake[k],unc_hists_fake[k],shapes,'fail',k,label)
                #if two_categories:
                #    plot_shape_nuisances(label, k, 'fail', compute_sf = compute_sf, compute_sf_onlynorm = compute_sf_onlynorm)
        
        save_yields(label, temp_hists)
        save_selection(label, preselection)
        save_weights(label, [k for k,v in samples.items()], weights)


# save reduced trees to produce datacards
# columns = ROOT.std.vector('string')()
# for ic in ['Q_sq', 'm_miss_sq', 'E_mu_star', 'E_mu_canc', 'Bmass']:
#     columns.push_back(ic)
# for k, v in samples.items():
#     v.Snapshot('tree', 'plots_ul/%s/tree_%s_datacard.root' %(label, k), columns)
