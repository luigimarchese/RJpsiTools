'''
This script does the final histograms for the fit for the rjpsi analysis
 - Computes the fakes from the fail region
 - Computes the shape uncertainties
 - Multiplies the veSsamplesnts for all the weights
 - Saves png, pdf and .root files with the histos in pass and fail regions + all the shape nuisances

Difference from _v7:
- deleted all the options to have different weights from different NN for the fakes: we decided to use the same NN for both contributions (data and MC)
Difference from _v136:- all the fakes differences
DIfference from _v12:
- Now its plots always the ch2 and ch4 flat fakerate, even in flatfakerate == False
- Deleted some options for jpsi_x_mu (like splitting into hm and lm)
Differnece from _v7:
- addition of hmlm_jpsi_x_mu_explicit option
  - for plots only
Difference from _v6:
- addition of renormalization with hammer weights from tau and mu
Difference from _v5:
- new option: add_hm_categories
  - it adds the high mass categories for the final fit
Difference from _v4:
- new option for jpsiXMu bkg to be splitted in the different contributions: 
   - FIXME: the datacard production gives an error: I will solve this when we have the new MC, because now we don't need that function
'''
'''
data = 65
mc = 65
alpha = 65

data_p03 = 67
mc_p03 = 67
alpha_p03 = 67

data_m03 = 66
mc_m03 = 66
alpha_m03 = 66
'''
'''
data = 3
mc = 3
alpha = 3
'''

data = 62
mc = 62
alpha = 62

data_p03 = 63
mc_p03 = 63
alpha_p03 = 63

data_m03 = 64
mc_m03 = 64
alpha_m03 = 64


#data = 31
#mc = 37
#alpha = 51

#system
import os
import copy
from datetime import datetime
import random
import time
import sys
import multiprocessing as mp
from argparse import ArgumentParser

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
from new_branches import to_define
from samples import weights, titles, colours, ff_weights
from selections import preselection, preselection_mc, pass_id, fail_id
from create_datacard_v3 import create_datacard_ch1, create_datacard_ch2, create_datacard_ch3, create_datacard_ch4, create_datacard_ch1_onlypass, create_datacard_ch3_onlypass
from plot_shape_nuisances_v4 import plot_shape_nuisances
from DiMuon import get_DiMuonBkgNorm, get_DiMuonBkg
from shape_comparison import shape_comparison

parser = ArgumentParser()

parser.add_argument('--asimov' ,default = False,action='store_true', help='Default makes no asimov')
parser.add_argument('--low_q2' ,default = False,action='store_true', help='Default makes high q2 histos')
parser.add_argument('--label', default='%s'%(datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')),help='addition to the preselection')
parser.add_argument('--preselection_plus', default='jpsivtx_svprob>1e-2',help='addition to the preselection')
parser.add_argument('--add_dimuon' ,default = False,action='store_true', help='Default doesnt add dimuon')
parser.add_argument('--compute_dimuon' ,default = False,action='store_true', help='Default doesnt compute dimuon; it works only if add_dimuon is True')
parser.add_argument('--dimuon_load', default='24Mar2022_15h29m26s',help='if add_dimuon== True and compute_dimuon==False, this is used to load the dimuon shapes from somewhere')

args = parser.parse_args()

label = args.label

preselection =  ' & '.join([preselection, args.preselection_plus])
preselection_mc =  ' & '.join([preselection_mc, args.preselection_plus])
print(preselection)

shape_nuisances = True
flat_fakerate = False # false mean that we use the NN weights for the fr

compute_sf_onlynorm = False # compute only the sf normalisation (best case)
blind_analysis = True
rjpsi = 1

add_dimuon = args.add_dimuon
compute_dimuon = args.compute_dimuon
dimuon_load = args.dimuon_load

asimov = args.asimov
print(type(asimov),asimov)
#asimov = False
only_pass = False

if asimov:
    blind_analysis=False
    rjpsi = 1

add_hm_categories = True #true if you want to add also the high mass categories to normalise the jpsimu bkg
jpsi_x_mu_split_jpsimother = True #true if you want to split the jpsimu bkg contributions depending on the jpsi mother
compress_xi_and_sigma = True # If jpsi_x_mu_split_jpsimother is True, this compress the xi and sigma contributes into 1 each
if args.low_q2:
    from histos import histos_lowq2 as histos_lm
else:
    from histos import histos as histos_lm

if jpsi_x_mu_split_jpsimother:
    if compress_xi_and_sigma:
        from samples import sample_names_explicit_jpsimother_compressed as sample_names
        from samples import basic_samples_names
        from samples import jpsi_x_mu_sample_jpsimother_splitting_compressed as jpsi_x_mu_samples
    else:
        from samples import  sample_names_explicit_jpsimother as sample_names
        from samples import  jpsi_x_mu_sample_jpsimother_splitting as jpsi_x_mu_samples

if add_hm_categories:
    from selections import preselection_hm, preselection_hm_mc
    from histos import histos_hm

dateTimeObj = datetime.now()
print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second, '.', dateTimeObj.microsecond)


ROOT.ROOT.EnableImplicitMT(mp.cpu_count())
ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

def make_directories(label):

    if not add_hm_categories:
        channels = ['ch1','ch2_flat']
        if not flat_fakerate:
            channels += ['ch2']
    else:
        channels = ['ch1','ch3','ch2_flat','ch4_flat']
        if not flat_fakerate:
            channels += ['ch2','ch4']
    
    print("Plots will be saved in %s"%label)
    for ch in channels:
        os.system('mkdir -p plots_ul/%s/%s/pdf/lin/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/pdf/log/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/png/lin/' %(label,ch))
        os.system('mkdir -p plots_ul/%s/%s/png/log/' %(label,ch))
    
    os.system('mkdir -p plots_ul/%s/datacards/' %label)
    if compute_dimuon:
        os.system('mkdir -p plots_ul/%s/dimuon/' %label)

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
        if jpsi_x_mu_split_jpsimother:
            if kk == 'jpsi_x_mu': continue
        if kk == 'dimuon':
            leg.AddEntry(temp_hists[k]['%s_%s' %(k, kk)], titles[kk], 'F' if kk!='data' else 'EP')
        elif kk == 'data' and asimov:
            continue
        else:
            leg.AddEntry(temp_hists[k]['%s_%s' %(k, kk)].GetValue(), titles[kk], 'F' if kk!='data' else 'EP')
            
    return leg

def create_datacard_prep(hists, shape_hists, shapes_names, sample_names, channel, name, label, which_sample_bbb_unc):
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
            if k == '%s_%s'%(name,isample):
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
            if k == '%s_%s'%(name,sname):
                hh = v.Clone()
                hh.SetName(sname + '_'+channel)
                hh.Write()

    if not add_dimuon or add_dimuon:
        if only_pass: #the rate of fakes must be == integral in case of only pass category fit, while ==1 in case of two regions
            if channel == 'ch1' :
                create_datacard_ch1_onlypass(label, name, myhists,False, jpsi_x_mu_samples, which_sample_bbb_unc)
            else:
                create_datacard_ch3_onlypass(label, name,  myhists, False, jpsi_x_mu_samples, which_sample_bbb_unc)

        else:
            if channel == 'ch1' :
                create_datacard_ch1(label, name,  myhists,  False, jpsi_x_mu_samples, which_sample_bbb_unc, add_dimuon = add_dimuon)
            elif channel == 'ch2' :
                create_datacard_ch2(label, name,  myhists,  False, jpsi_x_mu_samples, which_sample_bbb_unc)
            elif channel == 'ch3' :
                create_datacard_ch3(label, name,  myhists,  False, jpsi_x_mu_samples, which_sample_bbb_unc, add_dimuon = add_dimuon)
            else:
                create_datacard_ch4(label, name,  myhists,  False, jpsi_x_mu_samples, which_sample_bbb_unc)

    fout.Close()

# pass the jpsi_x_mu hists chi, sigma, lambda
def make_single_binbybin(hists, channel, label, name):
    if only_pass and (channel == 'ch2' or channel == 'ch4'):
        return
    fout = ROOT.TFile.Open('plots_ul/%s/datacards/datacard_%s_%s.root' %(label, channel, name), 'UPDATE')
    which_sample = []
    # loop over the bins of the hist
    for i in range(1,hists['sigma'].GetValue().GetNbinsX()+1):
        
        # if at least 2 of them are zero, no uncertainty bc gives problems to the fit
        flag = 0
        for s1,s2 in zip(['sigma','xi','lambdazero_b'],['xi','lambdazero_b','sigma']):
            if (hists[s1].GetValue().GetBinContent(i)<0.01 and hists[s2].GetValue().GetBinContent(i)<0.01):
                flag = 1
        if flag == 1:
            which_sample.append(None)
            continue
        # compute the quadratic sum of the stat unc of the 3 
        stat_unc = math.sqrt(hists['sigma'].GetValue().GetBinError(i)*hists['sigma'].GetValue().GetBinError(i)+hists['xi'].GetValue().GetBinError(i)*hists['xi'].GetValue().GetBinError(i)+ hists['lambdazero_b'].GetValue().GetBinError(i)*hists['lambdazero_b'].GetValue().GetBinError(i))
        #find the bin with highest uncertainty amongst the 3 contributes
        highest_stat = max(hists, key = lambda x:hists[x].GetValue().GetBinError(i))
        #print(i,hists['sigma'].GetValue().GetBinError(i),hists['xi'].GetValue().GetBinError(i),hists['lambdazero_b'].GetValue().GetBinError(i), stat_unc)
        which_sample.append(highest_stat)

        #define histo up and down
        histo_up = ROOT.TH1D('jpsi_x_mu_from_'+highest_stat+'_'+'jpsi_x_mu_from_'+highest_stat+'_single_bbb'+str(i)+channel+'Up_'+channel,'',hists[highest_stat].GetValue().GetNbinsX(),hists[highest_stat].GetValue().GetBinLowEdge(1), hists[highest_stat].GetValue().GetBinLowEdge(hists[highest_stat].GetValue().GetNbinsX() + 1))
        histo_down = ROOT.TH1D('jpsi_x_mu_from_'+highest_stat+'_'+'jpsi_x_mu_from_'+highest_stat+'_single_bbb'+str(i)+channel+'Down_'+channel,'',hists[highest_stat].GetValue().GetNbinsX(),hists[highest_stat].GetValue().GetBinLowEdge(1), hists[highest_stat].GetValue().GetBinLowEdge(hists[highest_stat].GetValue().GetNbinsX() + 1))

        for nbin in range(1,hists[highest_stat].GetValue().GetNbinsX()+1):
            if nbin == i:
                histo_up.SetBinContent(nbin,hists[highest_stat].GetValue().GetBinContent(nbin) + stat_unc)
                histo_up.SetBinError(nbin,stat_unc + math.sqrt(stat_unc))
                histo_down.SetBinContent(nbin,hists[highest_stat].GetValue().GetBinContent(nbin) - stat_unc)
                histo_down.SetBinError(nbin,stat_unc +math.sqrt( stat_unc))
            else:
                histo_up.SetBinContent(nbin,hists[highest_stat].GetValue().GetBinContent(nbin))
                histo_up.SetBinError(nbin,hists[highest_stat].GetValue().GetBinError(nbin))
                histo_down.SetBinContent(nbin,hists[highest_stat].GetValue().GetBinContent(nbin))
                histo_down.SetBinError(nbin,hists[highest_stat].GetValue().GetBinError(nbin))
        fout.cd()
        histo_up.Write()
        histo_down.Write()
    fout.Close()
    return which_sample


def make_binbybin(hist, sample, channel, label, name):
    if only_pass and (channel == 'ch2' or channel == 'ch4'):
        return

    fout = ROOT.TFile.Open('plots_ul/%s/datacards/datacard_%s_%s.root' %(label, channel, name), 'UPDATE')
    for i in range(1,hist.GetValue().GetNbinsX()+1):
        #histo_up = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+flag+'Up','jpsi_x_mu_bbb'+str(i)+flag+'Up',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        #histo_down = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+flag+'Down','jpsi_x_mu_bbb'+str(i)+flag+'Down',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        histo_up = ROOT.TH1D(sample+'_'+sample+'_bbb'+str(i)+channel+'Up_'+channel,'',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
        histo_down = ROOT.TH1D(sample+'_'+sample+'_bbb'+str(i)+channel+'Down_'+channel,'',hist.GetValue().GetNbinsX(),hist.GetValue().GetBinLowEdge(1), hist.GetValue().GetBinLowEdge(hist.GetValue().GetNbinsX() + 1))
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
    
    datacards = ['Q_sq','m_miss_sq','Bmass','jpsivtx_log10_lxy_sig_corr','jpsivtx_log10_lxy_sig','Q_sq_jpsimcorr','Bpt_reco']
    #datacards = histos

    # timestamp

    # create plot directories
    make_directories(label)
    
    #central_weights_string = 'br_weight'#*puWeight*sf_reco_total*sf_id_jpsi*sf_id_k'#*jpsimass_weights_for_correction*bc_mc_correction_weight_central' #the mc_correction_central weight is 1, added just to generalize the function for shape uncertainties

    central_weights_string = 'ctau_weight_central*br_weight*puWeight*sf_reco_total*sf_id_jpsi*sf_id_k*bc_mc_correction_weight_central_v2*jpsimass_weights_for_correction' #the mc_correction_central weight is 1, added just to generalize the function for shape uncertainties

    #central_weights_string = 'ctau_weight_central*br_weight*puWeight*sf_reco_total*sf_id_jpsi*sf_id_k*jpsimass_weights_for_correction'#*bc_mc_correction_weight_central' #the mc_correction_central weight is 1, added just to generalize the function for shape uncertainties

    # access the samples, via RDataFrames
    samples_orig = dict()
    samples_pres = dict()
    samples_lm = dict()

    tree_name = 'BTo3Mu'
    #tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/'

    print("=============================")
    print("====== Loading Samples ======")
    print("=============================")

    #load the samples (jpsi_x_mu even if I want it splitted)
    for k in sample_names:
        '''if k == 'data':
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_with_mc_corrections.root') 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/data_fakerate_only_iso.root') 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_withpres_withnn.root') 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_nopresel_withpresel_v1.root') 
        else:'''
        samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_nopresel_withpresel_v2_withnn_withidiso.root'%(tree_dir,k)) 
        #samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_nopresel.root'%(tree_dir,k)) 
        #samples_orig[k] = ROOT.RDataFrame(tree_name,'../samples/%s_nopresel_withpresel_v1.root'%(k)) 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_with_mc_corrections.root'%(tree_dir,k)) 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_nopresel_withpresel_v1.root'%(tree_dir,k)) 
            #samples_orig[k] = ROOT.RDataFrame(tree_name,'%s/%s_fakerate_only_iso.root'%(tree_dir,k)) 
        print(k)
    #Blind analysis: hide the value of rjpsi for the fit
    if blind_analysis:
        random.seed(2)
        rand = random.randint(0, 10000)
        blind = rand/10000 *1.5 +0.5
        #blind = 1.
    else:
        blind = 1.

    
    #################################################
    ####### Weights ################################
    #################################################

    '''
    correction_weight = 'mc_correction_gen_eta_weight'
    for k, v in samples_orig.items():
        #print(k)
        samples_orig[k] = samples_orig[k].Define('br_weight', '%f' %weights[k])
        if k=='jpsi_tau':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar*%f*%f*%s' %(blind,rjpsi,correction_weight))
        elif k=='jpsi_mu':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar*%s'%(correction_weight))
        elif 'jpsi_x_mu' in k: #works both if splitted or not
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*jpsimother_weight')
        elif k!='data' and 'jpsi_x_mu' not in k: # if Bc sample we need to multiply by correction weight
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*%s'%(correction_weight))            
        else:
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string  if k!='data' else 'br_weight') 
        '''
    print("weights definition")

    for k, v in samples_orig.items():
        print(k)
        #samples_orig[k] = samples_orig[k].Define('br_weight', '%f*iso_id_corr_weight_4' %weights[k])
        samples_orig[k] = samples_orig[k].Define('br_weight', '%f' %weights[k])
        if k=='jpsi_tau':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar*%f*%f' %(blind,rjpsi))
        elif k=='jpsi_mu':
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*hammer_bglvar')
        elif 'jpsi_x_mu' in k: #works both if splitted or not
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string +'*jpsimother_weight')
        else:
            samples_orig[k] = samples_orig[k].Define('tmp_weight', central_weights_string  if k!='data' else 'br_weight') 
            

        #define new columns   
        for new_column, new_definition in to_define: 
            if samples_orig[k].HasColumn(new_column):
                continue       
            samples_orig[k] = samples_orig[k].Define(new_column, new_definition)
    print("weights defined")
    if flat_fakerate == False:
        for sample in samples_orig:
            #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*nn/(
            #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_weight_w_weights_qsq_gen') 
            #if scale_mc_in_fail:
            #    if sample == 'data':
            #        samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_data') 
            #    else:
            #        samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_bcmu') 
            #else:
            

            samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data,alpha,mc,alpha))
            if sample in basic_samples_names and sample != 'data':
                samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr_p03', 'tmp_weight*1.3*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data_p03,alpha_p03,mc_p03,alpha_p03))
                samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr_m03', 'tmp_weight*0.7*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data_m03,alpha_m03,mc_m03,alpha_m03))
            else:
                samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr_p03', 'tmp_weight*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data_p03,alpha_p03,mc_p03,alpha_p03))
                samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr_m03', 'tmp_weight*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data_m03,alpha_m03,mc_m03,alpha_m03))

            
            #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*(fakerate_onlydata_%d)'%(data))
                #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_onlydata_13')
            #else:
            #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_onlymc_28')
            #    samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight*fakerate_onlymc_20 * (fakerate_alpha_42)/(1-fakerate_alpha_42)')
            #samples_orig[sample] = samples_orig[sample].Define('total_weight_wfr', 'tmp_weight')

    # the scale factor on the id on the third muon only for the PASS region
    for sample in samples_orig:
        #samples_orig[sample] = samples_orig[sample].Define('total_weight', 'tmp_weight*sf_id_k' if sample!='data' else 'tmp_weight')
        samples_orig[sample] = samples_orig[sample].Define('total_weight', 'tmp_weight' if sample!='data' else 'tmp_weight')
    print("preselections")
    ##############################################
    ##### Preselection ###########################
    ##############################################
    print("===================================")
    print("====== Applying Preselection ======")
    print("===================================")

    #Apply preselection for ch1 and ch2
    for k, v in samples_orig.items():
        print("Sample "+k )
        filter = preselection_mc if k!='data' else preselection
        samples_lm[k] = samples_orig[k].Filter(filter)
        #if scale_mc_in_fail:
        #    if k == 'data':
        #        continue
        #    if compute_mean_weights:
        #        mean_nn_data_weights_lm[k] = samples_lm[k].Mean("fakerate_data").GetValue()
        #        mean_nn_mc_weights_lm[k] = samples_lm[k].Mean("fakerate_bcmu").GetValue()
        #        print(mean_nn_data_weights_lm[k],mean_nn_mc_weights_lm[k])
        #    samples_lm[k] = samples_lm[k].Define('total_weight_wfr_norm', 'total_weight_wfr*%f/%f'%(mean_nn_data_weights_lm[k],mean_nn_mc_weights_lm[k]))
        #     #else:
        #        samples_lm[k] = samples_lm[k].Define('total_weight_wfr_norm', 'total_weight_wfr*%f/%f'%(mean_nn_data_lm[k].GetValue(),mean_nn_mc_weights_lm[k].GetValue()))
        
        #print("Sample "+k +" with "+str(samples_lm[k].Count().GetValue())+" events")
        
    histos_dictionaries = [histos_lm]
    #Apply preselection for ch3 and ch4 (high mass regions)
    if add_hm_categories:
        samples_hm = dict()
        #if scale_mc_in_fail and compute_mean_weights:
        #    mean_nn_data_weights_hm = dict()
        #    mean_nn_mc_weights_hm = dict()
        print("############################")
        for k, v in samples_orig.items():
            if not (k=='data' or 'jpsi_x_mu' in k): #only those samples are different from zero in the high mass region
                continue
            print("Sample "+k )
            filter = preselection_hm_mc if k!='data' else preselection_hm
            samples_hm[k] = samples_orig[k].Filter(filter)
            #print("Sample "+k +" with "+str(samples_hm[k].Count().GetValue())+" events")
            #if scale_mc_in_fail:
            #    if k == 'data':
            #        continue
            #    if compute_mean_weights:
            #        mean_nn_data_weights_hm[k] = samples_hm[k].Mean("fakerate_data").GetValue()
            #        mean_nn_mc_weights_hm[k] = samples_hm[k].Mean("fakerate_bcmu").GetValue()
            #        print(mean_nn_data_weights_hm[k],mean_nn_mc_weights_hm[k])
            #    samples_hm[k] = samples_hm[k].Define('total_weight_wfr_norm', 'total_weight_wfr*%f/%f'%(mean_nn_data_weights_hm[k],mean_nn_mc_weights_hm[k]))

        histos_dictionaries = [histos_lm, histos_hm]
    
    '''
    # request to divide jpsi_x_mu into all its contributions 
    if jpsi_x_mu_split:
        f_histo = ROOT.TFile.Open("decay_weight.root")
        histo = f_histo.Get('weight')
        jpsimother = {
            'other': histo.GetBinContent(1),
            'bzero': histo.GetBinContent(2),
            'bplus': histo.GetBinContent(3),
            'bzero_s': histo.GetBinContent(4),
            #'bplus_c': histo.GetBinContent(5),
            'sigmaminus_b': histo.GetBinContent(6),
            'lambdazero_b': histo.GetBinContent(7),
            'ximinus_b': histo.GetBinContent(8),
            'sigmazero_b': histo.GetBinContent(9),
            'xizero_b': histo.GetBinContent(10),
        }
        
        #division of jpsi_x_mu sample in different jpsi mother contributes
        for bkg_sample in jpsi_x_mu_sample_jpsimother_splitting:
            mother_name = bkg_sample.replace("jpsi_x_mu_from_","")
            
            filter_jpsi = ' & '.join(['jpsimother_weight == %s'%jpsimother[mother_name] ])
            # also split jpsi_x_mu for hm and lm if required
            if jpsi_x_mu_split_all or jpsi_x_mu_split_hmlm: 
                jpsi_x_mu_hmlm_dic = {'hm':'hmlm_flag == 0','lm':'hmlm_flag == 1'}
                for opt in jpsi_x_mu_hmlm_dic:
                    filter = ' & '.join([filter_jpsi,'%s'%(jpsi_x_mu_hmlm_dic[opt])])
                    samples_lm[bkg_sample + '_' + opt] = samples_lm['jpsi_x_mu'].Filter(filter) 
                    samples_hm[bkg_sample + '_' + opt] = samples_hm['jpsi_x_mu'].Filter(filter)
                    print("Splitting jpsi_x_mu in %s_%s; events %d for m<6.3 and %d for m>6.3"%(bkg_sample,opt,samples_lm[bkg_sample + '_' + opt].Count().GetValue(),samples_hm[bkg_sample + '_' + opt].Count().GetValue()))
                    #print("Splitting jpsi_x_mu in %s_%s"%(bkg_sample,opt))
                    
            else:
                print("Splitting jpsi_x_mu in %s"%bkg_sample)
                samples_lm[bkg_sample] = samples_lm['jpsi_x_mu'].Filter(filter_jpsi)
                #print("Splitting jpsi_x_mu in %s; events %d for m<6.3"%(bkg_sample,samples_lm[bkg_sample].Count().GetValue()))
                if add_hm_categories:
                    samples_hm[bkg_sample] = samples_hm['jpsi_x_mu'].Filter(filter_jpsi)
                    #print("Splitting jpsi_x_mu in %s; events %d for m>6.3"%(bkg_sample,samples_hm[bkg_sample].Count().GetValue()))

    '''
    samples_dictionaries = [samples_lm]
    if add_hm_categories:
        samples_dictionaries = [samples_lm, samples_hm]

    dateTimeObj = datetime.now()
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second, '.', dateTimeObj.microsecond)

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
                if ('jpsi_x_mu' not in sname and sname != 'data' ):    #Only Bc samples want this nuisance
                    shapes[sname + '_ctauUp'], shapes[sname + '_ctauDown'] = define_shape_nuisances(sname, shapes, samples, 'ctau', 'ctau_weight_central', 'ctau_weight_up', 'ctau_weight_down', central_weights_string)

                ###############################
                ########  PILE UP  ############
                ###############################

                if (sname != 'data'): # all MC samples
                    shapes[sname + '_puWeightUp'], shapes[sname + '_puWeightDown'] = define_shape_nuisances(sname,shapes, samples, 'puWeight', 'puWeight', 'puWeightUp', 'puWeightDown', central_weights_string)

                if compute_sf_onlynorm:
                    ###############################
                    ########  SF RECO  ############
                    ###############################
                    if (sname != 'data'):
                        shapes[sname + '_sfRecoUp'], shapes[sname + '_sfRecoDown'] = define_shape_nuisances(sname,shapes, samples,  'sfReco', 'sf_reco_total', 'sf_reco_all_up', 'sf_reco_all_down', central_weights_string)

                    ###############################
                    ########  SF ID  ##############
                    ###############################
		
                    # Only jpsi for now, bc the sf_id for the third muon is only in the pass region!
                    if (sname != 'data'):
                        shapes[sname + '_sfIdJpsiUp'], shapes[sname + '_sfIdJpsiDown'] = define_shape_nuisances(sname, shapes, samples, 'sfIdJpsi', 'sf_id_jpsi', 'sf_id_all_jpsi_up', 'sf_id_all_jpsi_down', central_weights_string)
                        shapes[sname + '_sfIdkUp'], shapes[sname + '_sfIdkDown'] = define_shape_nuisances(sname, shapes, samples, 'sfIdk', 'sf_id_k', 'sf_id_all_k_up', 'sf_id_all_k_down', central_weights_string)
            

                ######################################
                ########  MC CORRECTIONS  ############
                ######################################
                if ('jpsi_x_mu' not in sname and sname != 'data' ):    #Only Bc samples want this nuisance
                    shapes[sname + '_bccorrUp'], shapes[sname + '_bccorrDown'] = define_shape_nuisances(sname, shapes, samples, 'bccorr', 'bc_mc_correction_weight_central_v2', 'bc_mc_correction_weight_up_0p8_v2', 'bc_mc_correction_weight_down_0p8_v2', central_weights_string)
            
            
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
                    shapes['jpsi_mu_'+new_name] = shapes['jpsi_mu_'+new_name].Define('shape_weight_tmp', central_weights_string+'*'+ham+'*%f'%(ff_weights['jpsi_mu_'+new_name]/ff_weights['jpsi_mu']))
                    shapes['jpsi_tau_'+new_name] = samples['jpsi_tau']
                    shapes['jpsi_tau_'+new_name] = shapes['jpsi_tau_'+new_name].Define('shape_weight_tmp', central_weights_string+'*'+ham+'*%f'%(ff_weights['jpsi_tau_'+new_name]/ff_weights['jpsi_tau'])+'*%f*%f' %(blind,rjpsi))
                    

            if flat_fakerate == False:
                for name in shapes:
                    shapes[name] = shapes[name].Define('shape_weight_wfr','shape_weight_tmp*((fakerate_onlydata_%d-fakerate_alpha_%d*fakerate_onlymc_%d)/(1-fakerate_alpha_%d))'%(data,alpha,mc,alpha))
                    #shapes[name] = shapes[name].Define('shape_weight_wfr','shape_weight_tmp')

            for name in shapes:
                # this is to anticorrelate
                #if 'sfIdk' not in name:
                #    shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp*sf_id_k')
                shapes[name] = shapes[name].Define('shape_weight','shape_weight_tmp')
            


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
        if not flat_fakerate:
            temp_hists_fake_nn = {} # fail muon ID category
            temp_hists_fake_nn_p03 = {} # fail muon ID category
            temp_hists_fake_nn_m03 = {} # fail muon ID category
    
        for k, v in histos.items():    
            temp_hists     [k] = {}
            temp_hists_fake[k] = {}
            if not flat_fakerate:
                temp_hists_fake_nn[k] = {}
                if k in datacards+['jpsivtx_log10_lxy_sig_corr'] :
                    temp_hists_fake_nn_p03[k] = {}
                    temp_hists_fake_nn_m03[k] = {}
            for kk, vv in samples.items():
                temp_hists     [k]['%s_%s' %(k, kk)] = vv.Filter(pass_id).Histo1D(v[0], k, 'total_weight')
                temp_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight')
                if not flat_fakerate:
                    #if scale_mc_in_fail:
                    #    if kk == 'data':
                    #        temp_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr')
                    #    else:
                    #        temp_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr_norm')
                    #else:
                    temp_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr')
                    if k in datacards+['jpsivtx_log10_lxy_sig_corr']:
                        temp_hists_fake_nn_p03[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr_p03')
                        temp_hists_fake_nn_m03[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'total_weight_wfr_m03')

            # Di muon bkg
            #print("CIAO",add_dimuon,iteration,k)
            if add_dimuon: #changed with moving if k ==  'Q_sq':  later, 15_03_2022
                if not iteration:
                    if k == 'Q_sq': #changed 15_03_2022
                        if compute_dimuon:
                            print("Doing the Dimuon",k)
                            Norm_SRloose = get_DiMuonBkgNorm() 
                            temp_hists[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, pass_id+" & Bmass<6.3 & Q_sq>5.5 &"+args.preselection_plus, 0, 0, label, 'ch1').GetValue()
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, fail_id+" & Bmass<6.3 & Q_sq>5.5 & "+args.preselection_plus, 0, 0, label, 'ch2_flat').GetValue()
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, fail_id+" & Bmass<6.3 & Q_sq>5.5 &"+args.preselection_plus, 0, 1, label, 'ch2').GetValue()
                            '''
                            #save them on file
                            fout = ROOT.TFile.Open('plots_ul/%s/dimuon/dimuon_%s.root' %(label, k), 'UPDATE')
                            fout.cd()
                            temp_hists[k]['%s_dimuon'%k].SetName("dimuon_ch1")
                            temp_hists[k]['%s_dimuon'%k].Write()
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k].SetName("dimuon_ch2_flat")
                            temp_hists_fake[k]['%s_dimuon'%k].Write()
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k].SetName("dimuon_ch2")
                                temp_hists_fake_nn[k]['%s_dimuon'%k].Write()
                            '''
                            fout.Close()
                        #take from file
                        else:
                            dimuon_path = 'plots_ul/'+dimuon_load+'/dimuon/'
                            fdimuon = ROOT.TFile.Open(dimuon_path+"/dimuon_Q_sq.root","r")
                            #a = ROOT.RDF.RResultPtr[ROOT.TH1D]()
                            #a.GetValue() = f.Get("dimuon_ch1")
                            #print(a,type(a))
                            temp_hists[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch1")
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch2_flat")
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch2")
                            '''
                            print(temp_hists[k]['%s_dimuon'%k])
                if iteration:
                    if k == 'jpsivtx_log10_lxy_sig_corr':  #changed from this line 15_03_2022 up to
                        if compute_dimuon:
                            print("Doing the Dimuon",k)
                            temp_hists[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, pass_id+" & Bmass>6.3 &"+args.preselection_plus, 5, 0, label, 'ch3').GetValue()
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, fail_id+" & Bmass>6.3 &"+args.preselection_plus, 5, 0, label, 'ch4_flat').GetValue()
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k] = get_DiMuonBkg(Norm_SRloose, fail_id+" & Bmass>6.3 &"+args.preselection_plus, 5, 1, label, 'ch4').GetValue()  
                            '''
                            #save them on file
                            fout = ROOT.TFile.Open('plots_ul/%s/dimuon/dimuon_%s.root' %(label, k), 'UPDATE')
                            fout.cd()
                            temp_hists[k]['%s_dimuon'%k].SetName("dimuon_ch3")
                            temp_hists[k]['%s_dimuon'%k].Write()
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k].SetName("dimuon_ch4_flat")
                            temp_hists_fake[k]['%s_dimuon'%k].Write()
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k].SetName("dimuon_ch4")
                                temp_hists_fake_nn[k]['%s_dimuon'%k].Write()
                            '''
                            fout.Close()
                        #take from file
                        else:
                            dimuon_path = 'plots_ul/'+dimuon_load+'/dimuon/'
                            fdimuon = ROOT.TFile.Open(dimuon_path+"/dimuon_jpsivtx_log10_lxy_sig_corr.root","r")
                            #a = ROOT.RDF.RResultPtr[ROOT.TH1D]()
                            #a.GetValue() = f.Get("dimuon_ch1")
                            #print(a,type(a))
                            temp_hists[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch3")
                            '''
                            temp_hists_fake[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch4_flat")
                            if not flat_fakerate:
                                temp_hists_fake_nn[k]['%s_dimuon'%k] = fdimuon.Get("dimuon_ch4")
                            '''
                            print(temp_hists[k]['%s_dimuon'%k])
                            #f.Close()

                        #else:
                        #take it from a file
                        
                '''if iteration:
                    temp_hists[k]['%s_dimuon'%k] = get_DiMuonBkg(pass_id+" & Bmass>6.3", 0)
                    print(type(temp_hists[k]['%s_dimuon'%k]))
                    print(type(temp_hists[k]['%s_jpsi_tau'%k]))
                    print(temp_hists[k].items())
                    temp_hists_fake[k]['%s_dimuon'%k] = get_DiMuonBkg(fail_id+" & Bmass>6.3", 0)
                    if not flat_fakerate:
                temp_hists_fake_nn[k]['%s_dimuon'%k] = get_DiMuonBkg(fail_id+" & Bmass>6.3", 0)'''
                #print(type(temp_hists[k]['%s_dimuon'%k]))
                #print(type(temp_hists[k]['%s_jpsi_tau'%k]))
                #print(temp_hists[k].items())

    
        # Create pointers for the shapes histos 
        if shape_nuisances:
            print('====> shape uncertainties histos')
            unc_hists      = {} # pass muon ID category
            unc_hists_fake = {} # pass muon ID category
            if not flat_fakerate:
                unc_hists_fake_nn = {} # pass muon ID category
            for k, v in histos.items():    
                # Compute them only for the variables that we want to fit
                if (k not in datacards and iteration == 0) or (k not in histos and iteration):
                    #if (k not in datacards and iteration == 0) or (k!='Bmass' and iteration):
                    continue
                unc_hists     [k] = {}
                unc_hists_fake[k] = {}
                if not flat_fakerate:
                    unc_hists_fake_nn[k] = {}
                for kk, vv in shapes.items():
                    unc_hists     [k]['%s_%s' %(k, kk)] = vv.Filter(pass_id).Histo1D(v[0], k, 'shape_weight')
                    unc_hists_fake[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight')
                    if not flat_fakerate:
                        #if scale_mc_in_fail:
                        #    if kk == 'data':
                        #        unc_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight_wfr')
                        #    else:
                        #        unc_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight_wfr_norm')
                        #else:
                        unc_hists_fake_nn[k]['%s_%s' %(k, kk)] = vv.Filter(fail_id).Histo1D(v[0], k, 'shape_weight_wfr')
                                
        
        print('====> now looping')
        for k, v in histos.items():
            print("Histo %s"%k)
            single_bbb_histos = {}
            single_bbb_histos_fake = {}
            for sample,sample_item in samples.items():
                if "jpsi_x_mu" in sample:
                    if not jpsi_x_mu_split_jpsimother: # The general binbybin only for jpsi_x_mu when it is not splitted
                        make_binbybin(temp_hists[k]['%s_%s'%(k,sample)],sample,channels[0], label, k)
                        if not flat_fakerate:
                            make_binbybin(temp_hists_fake_nn[k]['%s_%s'%(k,sample)],sample,channels[1], label, k)
                        else:
                            make_binbybin(temp_hists_fake[k]['%s_%s'%(k,sample)],sample,channels[1], label, k)
                    if 'sigma' in sample or 'xi' in sample or 'lambda' in sample:
                        single_bbb_histos[sample.replace("jpsi_x_mu_from_","")]=temp_hists[k]['%s_%s'%(k,sample)]
                        if not flat_fakerate:
                            single_bbb_histos_fake[sample.replace("jpsi_x_mu_from_","")]=temp_hists_fake_nn[k]['%s_%s'%(k,sample)]
                        else:
                            single_bbb_histos_fake[sample.replace("jpsi_x_mu_from_","")]=temp_hists_fake[k]['%s_%s'%(k,sample)]

            which_sample_bbb_unc = make_single_binbybin(single_bbb_histos, channels[0], label, k)
            which_sample_bbb_unc_fake = make_single_binbybin(single_bbb_histos_fake, channels[1], label, k)
            
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

            if not flat_fakerate:
                for i, kv in enumerate(temp_hists_fake_nn[k].items()):
                    key = kv[0]
                    ihist = kv[1]
                    sample_name = key.split(k+'_')[1]
                    for i in range(1,ihist.GetNbinsX()+1):
                        if ihist.GetBinContent(i) <= 0:
                            ihist.SetBinContent(i,0.0001)

            if shape_nuisances and ((k in datacards and  iteration==0) or (k in histos and iteration)):
            #if shape_nuisances and ((iteration==0) or (k == 'Bmass' and iteration)):
                
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
                
                if not flat_fakerate:
                    for i, kv in enumerate(unc_hists_fake_nn[k].items()):
                        key = kv[0]
                        ihist = kv[1]
                        sample_name = key.split(k+'_')[1]
                        for i in range(1,ihist.GetNbinsX()+1):
                            if ihist.GetBinContent(i) <= 0:
                                ihist.SetBinContent(i,0.0001)
                    

            c1.cd()
            # add also comb bkg histos
            if add_dimuon and ((not iteration and k=='Q_sq') or (iteration and k=='jpsivtx_log10_lxy_sig_corr')):
                samples_for_legend = [str(k) for k in samples]+['dimuon']
            else:
                samples_for_legend = [str(k) for k in samples]

            #print("CIAO2",k,temp_hists, samples_for_legend)
            leg = create_legend(temp_hists, samples_for_legend, titles)
            main_pad.cd()
            main_pad.SetLogy(False)
        
            # some look features
            maxima = [] 
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
            ths1      = ROOT.THStack('stack', '') #what I want to show
            ths1_fake = ROOT.THStack('stack_fake', '')
            if not flat_fakerate:
                ths1_fake_nn = ROOT.THStack('stack_fake_nn', '')

            for i, kv in enumerate(temp_hists[k].items()):
                
                key = kv[0]
                if key=='%s_data'%k: continue
                ihist = kv[1]
                ihist.SetMaximum(1.6*max(maxima))
                ihist.Draw('hist' + 'same'*(i>0))
                #print("Integral %s %f"%(key,ihist.Integral()))
                if not jpsi_x_mu_split_jpsimother:
                    if key=='%s_dimuon':
                        ths1.Add(ihist)
                    else:
                        ths1.Add(ihist.GetValue())

                else:
                    # if I want to explicitly see the splitting in the plots, I save them in ths1
                    #if jpsi_x_mu_explicit_show_on_plots: 
                    if key=='%s_jpsi_x_mu'%k: continue
                    if key=='%s_dimuon'%k:
                        ths1.Add(ihist)
                    else:
                        ths1.Add(ihist.GetValue())
            
            # apply same aestethics to pass and fail
            #print(temp_hists_fake[k])
            for kk in temp_hists_fake[k].keys():
                temp_hists_fake[k][kk].GetXaxis().SetTitle(temp_hists[k][kk].GetXaxis().GetTitle())
                temp_hists_fake[k][kk].GetYaxis().SetTitle(temp_hists[k][kk].GetYaxis().GetTitle())
                temp_hists_fake[k][kk].SetLineColor(temp_hists[k][kk].GetLineColor())
                temp_hists_fake[k][kk].SetFillColor(temp_hists[k][kk].GetFillColor())

            if not flat_fakerate:
                for kk in temp_hists_fake[k].keys():
                    temp_hists_fake_nn[k][kk].GetXaxis().SetTitle(temp_hists[k][kk].GetXaxis().GetTitle())
                    temp_hists_fake_nn[k][kk].GetYaxis().SetTitle(temp_hists[k][kk].GetYaxis().GetTitle())
                    temp_hists_fake_nn[k][kk].SetLineColor(temp_hists[k][kk].GetLineColor())
                    temp_hists_fake_nn[k][kk].SetFillColor(temp_hists[k][kk].GetFillColor())
                

            # fakes for the fail contribution
            # subtract data to MC
            #if not flat_fakerate:
            #    temp_hists[k]['%s_fakes' %k] = temp_hists_fake[k]['%s_data' %k].Clone()
            #else:
            #    temp_hists[k]['%s_fakes' %k] = temp_hists_fake_nn[k]['%s_data' %k].Clone()
            if not flat_fakerate:
                temp_hists[k]['%s_fakes' %k] = temp_hists_fake_nn[k]['%s_data' %k].Clone()
                fakes_failnn = temp_hists[k]['%s_fakes' %k]
                fakes_fail = temp_hists_fake[k]['%s_data' %k].Clone()
            else:
                temp_hists[k]['%s_fakes' %k] = temp_hists_fake[k]['%s_data' %k].Clone()
                fakes_fail = temp_hists[k]['%s_fakes' %k]
            
            # Subtract to fakes all the contributions of other samples in the fail region
            
            #fakes from fail
            # FIXME

            for i, kv in enumerate(temp_hists_fake[k].items()):
                if 'data' in kv[0]:
                    kv[1].SetLineColor(ROOT.kBlack)
                    continue
                    #elif 'jpsi_x_mu_' in kv[0]: #if one of the splittings of jpsi_x_mu
                    #    continue
                else:
                    if 'dimuon' in kv[0]:
                        fakes_fail.Add(kv[1], -1.)
                    else:
                        fakes_fail.Add(kv[1].GetPtr(), -1.)

            # fakes from fail *NN
            if not flat_fakerate:
                for i, kv in enumerate(temp_hists_fake_nn[k].items()):
                    if 'data' in kv[0]:
                        kv[1].SetLineColor(ROOT.kBlack)
                        continue
                    else:
                        if 'dimuon'in kv[0]:
                            fakes_failnn.Add(kv[1], -1.)
                        else:
                            fakes_failnn.Add(kv[1].GetPtr(), -1.)
                            
            if shape_nuisances and ((k in datacards and  iteration==0) or (k =='jpsivtx_log10_lxy_sig_corr' and iteration)):                
                for i, kv in enumerate(temp_hists_fake_nn_p03[k].items()):
                    if 'data' in kv[0]:
                        continue
                    else:
                        temp_hists_fake_nn_p03[k]['%s_data' %k].Add(kv[1].GetPtr(), -1.)

                for i, kv in enumerate(temp_hists_fake_nn_m03[k].items()):
                    if 'data' in kv[0]:
                        continue
                    else:
                        temp_hists_fake_nn_m03[k]['%s_data' %k].Add(kv[1].GetPtr(), -1.)


            # choose which one goes to Pass region
            if not flat_fakerate:
                #check fakes do not have <= 0 bins
                for b in range(1,fakes_failnn.GetNbinsX()+1):
                    if fakes_failnn.GetBinContent(b)<=0.:
                        fakes_failnn.SetBinContent(b,0.0001)
                fakes_failnn.SetFillColor(colours['fakes'])
                fakes_failnn.SetFillStyle(1001)
                fakes_failnn.SetLineColor(colours['fakes'])
                fakes = fakes_failnn.Clone()

            for b in range(1,fakes_fail.GetNbinsX()+1):
                if fakes_fail.GetBinContent(b)<=0.:
                    fakes_fail.SetBinContent(b,0.0001)
            fakes_fail.SetFillColor(colours['fakes'])
            fakes_fail.SetFillStyle(1001)
            fakes_fail.SetLineColor(colours['fakes'])
            
            if flat_fakerate:
                fakes = fakes_fail.Clone()

            #fakes_forfail = fakes.Clone()
            if flat_fakerate:
                fakes.Scale(weights['fakes'])
            fakes.Scale(weights['fakes'])
            #fakes.Scale(0.7)

            #temp_hists[k]['%s_fakes' %k] = fakes
            ths1.Add(fakes)
            
                
            #print(k,fakes.Integral())
            maxima.append(fakes.GetMaximum())
            ths1.Draw('hist')
            try:
                ths1.GetXaxis().SetTitle(v[1])
            except:
                continue
            ths1.GetYaxis().SetTitle('events')
            ths1.SetMaximum(1.6*max(sum(maxima), data_max))
            ths1.SetMinimum(0.0001)

        
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
                
            CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = 'L = 59.7 fb^{-1}')
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
            ratio_stats.SetMinimum(0.0001) # and this is for symmetry
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

                if not jpsi_x_mu_split_jpsimother:
                    norm_stack.Add(hh)
                else:
                    # if I want to explicitly see the splitting in the plots
                    #if jpsi_x_mu_explicit_show_on_plots: 
                    if kk=='%s_jpsi_x_mu'%k: continue
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
        
            if shape_nuisances and ((k in datacards and  iteration==0) or (k =='jpsivtx_log10_lxy_sig_corr' and iteration)):
            #if shape_nuisances and ((iteration==0) or (k == 'Bmass' and iteration)):
                temp_hists_fake_nn_p03[k]['%s_data'%k].Scale(fakes.Integral()/temp_hists_fake_nn_p03[k]['%s_data'%k].Integral())
                unc_hists[k]['%s_fakes_fakesshapeUp'%k] = temp_hists_fake_nn_p03[k]['%s_data'%k]

                temp_hists_fake_nn_m03[k]['%s_data'%k].Scale(fakes.Integral()/temp_hists_fake_nn_m03[k]['%s_data'%k].Integral())
                unc_hists[k]['%s_fakes_fakesshapeDown'%k] = temp_hists_fake_nn_m03[k]['%s_data'%k]

                shapes['fakes_fakesshapeUp'] = [] #just for the name
                shapes['fakes_fakesshapeDown'] = []

                create_datacard_prep(temp_hists[k], unc_hists[k], shapes, samples_for_legend, channels[0], k, label, which_sample_bbb_unc)
                #shape_comparison(label, k, channels[0], [name for name,v in samples.items()], verbose = True)
                if not add_dimuon:
                    plot_shape_nuisances(label, k, channels[0], [name for name,v in samples.items()], which_sample_bbb_unc, compute_sf = False, compute_sf_onlynorm = compute_sf_onlynorm)
                    # script per comparison shapes tau mu fakes

            #####################################################
            # Now creating and saving the stack of the fail region

            if not flat_fakerate:
                c1.cd()
                main_pad.cd()
                main_pad.SetLogy(False)
                max_fake = []
                for i, kv in enumerate(temp_hists_fake_nn[k].items()):
                    key = kv[0]
                    if key=='%s_data'%k: 
                        max_fake.append(kv[1].GetMaximum())
                        continue
                    ihist = kv[1]
                    #print("Integral %s %f"%(key,ihist.Integral()))
                    if not jpsi_x_mu_split_jpsimother:
                        if key=='%s_dimuon'%k:
                            ths1_fake_nn.Add(ihist)
                        else:
                            ths1_fake_nn.Add(ihist.GetValue())
                    else:
                        # if I want to explicitly see the splitting in the plots, I save them in ths1_fake_nn
                        if key=='%s_jpsi_x_mu'%k: continue
                        if key=='%s_dimuon'%k:
                            ths1_fake_nn.Add(ihist)
                        else:
                            ths1_fake_nn.Add(ihist.GetValue())

                temp_hists_fake_nn[k]['%s_fakes' %k] = fakes_failnn.Clone()
                ths1_fake_nn.Add(fakes_failnn.Clone())
                ths1_fake_nn.Draw('hist')
                ths1_fake_nn.SetMaximum(2.*sum(max_fake))
                ths1_fake_nn.SetMinimum(0.0001)
                ths1_fake_nn.GetYaxis().SetTitle('events')
                
                stats_fake = ths1_fake_nn.GetStack().Last().Clone()
                stats_fake.SetLineColor(0)
                stats_fake.SetFillColor(ROOT.kGray+1)
                stats_fake.SetFillStyle(3344)
                stats_fake.SetMarkerSize(0)
                stats_fake.Draw('E2 SAME')
                
                if not asimov:
                    temp_hists_fake_nn[k]['%s_data'%k].Draw('EP SAME')
                CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

                leg = create_legend(temp_hists, [str(k) for k in samples], titles)
                if flat_fakerate:
                    leg.AddEntry(fakes, 'fakes flat', 'F')    
                else:
                    leg.AddEntry(fakes, 'fakes nn', 'F')    
                leg.AddEntry(stats, 'stat. unc.', 'F')
                leg.Draw('same')

                # Ratio for pass region
                
                ratio_pad.cd()
                ratio_fake = temp_hists_fake_nn[k]['%s_data'%k].Clone()
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
                
                for kk, vv in temp_hists_fake_nn[k].items():
                    if 'data' in kk: continue
                    hh = vv.Clone()
                    hh.Divide(stats_fake)
                    if not jpsi_x_mu_split_jpsimother:
                        norm_stack_fake.Add(hh)
                    else:
                        # if I want to explicitly see the splitting in the plots
                        if kk=='%s_jpsi_x_mu'%k: continue
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
                
                ths1_fake_nn.SetMaximum(20*max(sum(maxima), data_max))
                ths1_fake_nn.SetMinimum(10)
                main_pad.SetLogy(True)
                c1.Modified()
                c1.Update()
                
                c1.SaveAs('plots_ul/%s/%s/pdf/log/%s.pdf' %(label, channels[1], k))
                c1.SaveAs('plots_ul/%s/%s/png/log/%s.png' %(label, channels[1], k))

                if not flat_fakerate:
                    if shape_nuisances and ((k in datacards and  iteration==0) or (k in histos and iteration)):
                    #if shape_nuisances and ((iteration==0) or (k == 'Bmass' and iteration)):

                        if not flat_fakerate:
                            create_datacard_prep(temp_hists_fake_nn[k], unc_hists_fake_nn[k], shapes, samples_for_legend, channels[1], k, label, which_sample_bbb_unc_fake)
                        else:
                            create_datacard_prep(temp_hists_fake_nn[k], unc_hists_fake[k], shapes, samples_for_legend, channels[1], k, label, which_sample_bbb_unc_fake)
                        #create_datacard_prep(temp_hists_fake[k],unc_hists_fake[k],shapes,'fail',k,label)
                        #shape_comparison(label, k, channels[1], [name for name,v in samples.items()], verbose = True)
                        if not only_pass and not add_dimuon:
                            plot_shape_nuisances(label, k, channels[1], [name for name,v in samples.items()], which_sample_bbb_unc_fake, compute_sf = False, compute_sf_onlynorm = compute_sf_onlynorm)
            #####################################################
            # Now creating and saving the stack of the fail region

            c1.cd()
            main_pad.cd()
            main_pad.SetLogy(False)
            max_fake = []
            for i, kv in enumerate(temp_hists_fake[k].items()):
                key = kv[0]
                if key=='%s_data'%k: 
                    max_fake.append(kv[1].GetMaximum())
                    continue
                ihist = kv[1]
                #print("Integral %s %f"%(key,ihist.Integral()))
                if not jpsi_x_mu_split_jpsimother:
                    if key=='%s_dimuon'%k:
                        ths1_fake.Add(ihist)
                    else:
                        ths1_fake.Add(ihist.GetValue())
                else:
                    # if I want to explicitly see the splitting in the plots, I save them in ths1_fake
                    if key=='%s_jpsi_x_mu'%k: continue
                    if key=='%s_dimuon'%k:
                        ths1_fake.Add(ihist)
                    else:
                        ths1_fake.Add(ihist.GetValue())

            temp_hists_fake[k]['%s_fakes' %k] = fakes_fail.Clone()
            ths1_fake.Add(fakes_fail.Clone())
            ths1_fake.Draw('hist')
            ths1_fake.SetMaximum(2.*sum(max_fake))
            ths1_fake.SetMinimum(0.0001)
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
                if not jpsi_x_mu_split_jpsimother:
                    norm_stack_fake.Add(hh)
                else:
                    # if I want to explicitly see the splitting in the plots
                    if kk=='%s_jpsi_x_mu'%k: continue
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

            c1.SaveAs('plots_ul/%s/%s_flat/pdf/lin/%s.pdf' %(label, channels[1], k))
            c1.SaveAs('plots_ul/%s/%s_flat/png/lin/%s.png' %(label, channels[1], k))

            ths1_fake.SetMaximum(20*max(sum(maxima), data_max))
            ths1_fake.SetMinimum(10)
            main_pad.SetLogy(True)
            c1.Modified()
            c1.Update()

            c1.SaveAs('plots_ul/%s/%s_flat/pdf/log/%s.pdf' %(label, channels[1], k))
            c1.SaveAs('plots_ul/%s/%s_flat/png/log/%s.png' %(label, channels[1], k))

            if flat_fakerate:
                if shape_nuisances and ((k in datacards and  iteration==0) or (k in histos and iteration)):
                #if shape_nuisances and ((iteration==0) or (k == 'Bmass' and iteration)):

                    if not flat_fakerate:
                        create_datacard_prep(temp_hists_fake[k], unc_hists_fake_nn[k], shapes, [name for name,v in samples.items()], channels[1], k, label, which_sample_bbb_unc_fake)
                    else:
                        create_datacard_prep(temp_hists_fake[k], unc_hists_fake[k], shapes, [name for name,v in samples.items()], channels[1], k, label, which_sample_bbb_unc_fake)
                        
                    #create_datacard_prep(temp_hists_fake[k],unc_hists_fake[k],shapes,'fail',k,label)
                    #shape_comparison(label, k, channels[1], [name for name,v in samples.items()], verbose = True)
                    if not only_pass and not add_dimuon:
                        plot_shape_nuisances(label, k, channels[1], [name for name,v in samples.items()], which_sample_bbb_unc_fake, compute_sf = False, compute_sf_onlynorm = compute_sf_onlynorm)

            if channels[0] == 'ch1' and not flat_fakerate:
                shape_comparison({'jpsi_mu':temp_hists[k]['%s_jpsi_mu' %k],'jpsi_tau':temp_hists[k]['%s_jpsi_tau' %k],"fakes":fakes},label, k, channels[0], [name for name,v in samples.items()], verbose = True)
            if channels[1] == 'ch2' and not flat_fakerate:
                shape_comparison({'jpsi_mu':temp_hists_fake_nn[k]['%s_jpsi_mu' %k],'jpsi_tau':temp_hists_fake_nn[k]['%s_jpsi_tau' %k],"fakes":fakes_failnn},label, k, channels[1], [name for name,v in samples.items()], verbose = True)
            
            #try:
            #    fdimuon.Close()


save_yields(label, temp_hists)
save_selection(label, preselection)
save_weights(label, [k for k,v in samples.items()], weights)

dateTimeObj = datetime.now()
print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second, '.', dateTimeObj.microsecond)


# save reduced trees to produce datacards
# columns = ROOT.std.vector('string')()
# for ic in ['Q_sq', 'm_miss_sq', 'E_mu_star', 'E_mu_canc', 'Bmass']:
#     columns.push_back(ic)
# for k, v in samples.items():
#     v.Snapshot('tree', 'plots_ul/%s/tree_%s_datacard.root' %(label, k), columns)
