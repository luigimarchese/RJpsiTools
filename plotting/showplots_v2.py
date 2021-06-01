import os
import copy
import ROOT
import numpy as np
from datetime import datetime
from bokeh.palettes import viridis, all_palettes
import random
import time
from array import array
import pickle

from histos import histos
from cmsstyle import CMS_lumi
from new_branches import to_define
from samples import weights, sample_names, titles
from selections import preselection, preselection_mc, pass_id, fail_id
from officialStyle import officialStyle
from create_datacard import create_datacard_pass,create_datacard_fail
from plot_shape_nuisances import plot_shape_nuisances

from keras.models import load_model

shape_nuisances = True
flat_fakerate = False
blind_analysis = False
rjpsi = 1

start_time = time.time()

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

def make_directories(label):

    os.system('mkdir -p plots_ul/%s/pdf/lin/' %label)
    os.system('mkdir -p plots_ul/%s/pdf/log/' %label)
    os.system('mkdir -p plots_ul/%s/png/lin/' %label)
    os.system('mkdir -p plots_ul/%s/png/log/' %label)

    if flat_fakerate:
        os.system('mkdir -p plots_ul/%s/fail_region/pdf/lin/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region/pdf/log/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region/png/lin/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region/png/log/' %label)
    else:
        os.system('mkdir -p plots_ul/%s/fail_region_reweight/pdf/lin/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region_reweight/pdf/log/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region_reweight/png/lin/' %label)
        os.system('mkdir -p plots_ul/%s/fail_region_reweight/png/log/' %label)

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

def create_datacard_prep(hists,shape_hists,shapes_names,flag,name,label):
    '''
    Creates and saves the root file with the histograms of each contribution.
    Saves the histograms of the shape nuisances.
    Calls the 'create datacard' function, both for the pass and fail regions, 
    to write the text datacard for the fit in combine. 
    '''
    fout = ROOT.TFile.Open('plots_ul/%s/datacards/datacard_%s_%s.root' %(label,flag, name), 'recreate')
    fout.cd()
    myhists = dict()
    for k, v in hists.items():
        for isample in sample_names + ['fakes']:
            if isample in k:
                hh = v.Clone()
                if isample == 'data':
                    hh.SetName(isample+'_obs')
                else:
                    hh.SetName(isample)
                hh.Write()
                myhists[isample] = hh.Clone()
        
    # Creates the shape nuisances both for Pass and Fail regions
    for k,v in shape_hists.items():
        for sname in shapes_names:
            if sname in k:
                hh = v.Clone()
                hh.SetName(sname)
                hh.Write()
    # datacard txt are different depending on the region
    if flag == 'pass':
        create_datacard_pass(myhists,name, label)
    else:
        create_datacard_fail(myhists,name, label)
    fout.Close()

                
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
    
    datacards = ['mu1pt', 'Q_sq', 'm_miss_sq', 'E_mu_star', 'E_mu_canc', 'bdt_tau', 'Bmass', 'mcorr', 'decay_time_ps','k_raw_db_corr_iso04_rel']
    
    # timestamp
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

    # create plot directories
    make_directories(label)
    
    # access the samples, via RDataFrames
    samples = dict()

    tree_name = 'BTo3Mu'
    #Different paths depending on the sample
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'

    '''tree_dir_data = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #data
    tree_dir_hb = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #Hb
    tree_dir_bc = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Apr14' #Bc 
    tree_dir_psitau = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar25' #psi2stau
    tree_hbmu = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar16' #Hb mu filter
    '''

    for k in sample_names:
        samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_fakerate.root'%(tree_dir,k))
    
    print("=============================")
    print("====== Samples loaded =======")
    print("=============================")

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
    for k, v in samples.items():
        samples[k] = samples[k].Define('br_weight', '%f' %weights[k])
        #for jpsi tau apply ctau, pu and ff weights. Plus the values for the blind analyss and rjpsi
        if k=='jpsi_tau':
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*%f*%f' %(blind,rjpsi))
            #samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight*hammer_bglvar*%f*%f' %(blind,rjpsi))
        # jpsi mu apply ctau, pu and ff weights
        elif k=='jpsi_mu':
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar')
            #samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight*hammer_bglvar')
        #For all the other samples we apply ctau and pu
        #For the Bc samples the ctau contribution is != 1., while for the background it is ==1
        else:
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight' if k!='data' else 'br_weight') 
            #samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight' if k!='data' else 'br_weight') 

        #define new columns
        for new_column, new_definition in to_define: 
            if samples[k].HasColumn(new_column):
                continue
            samples[k] = samples[k].Define(new_column, new_definition)

    if flat_fakerate == False:
        for sample in samples:
            samples[sample] = samples[sample].Define('total_weight_wfr', 'total_weight*nn/(1-nn)') 

    #Apply preselection. 
    for k, v in samples.items():
        filter = preselection_mc if k!='data' else preselection
        samples[k] = samples[k].Filter(filter)

    # Shape nuisances definition
    # Create a new disctionary "shapes", similar to the "samples" one defined for the datasets
    # Each entry of the dic is a nuisance for a different dataset
    if shape_nuisances :
        shapes = dict()
        #ctau nuisances
        for sname in samples:
            #Only Bc samples want this nuisance
            if (sname != 'jpsi_x_mu' and sname != 'data' ):
                shapes[sname + '_ctauUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight*hammer_bglvar')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight*hammer_bglvar*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight')
                shapes[sname + '_ctauDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight*hammer_bglvar')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight*hammer_bglvar*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight')
        
            # Pile up nuisances
            if (sname != 'data'):
                shapes[sname + '_puWeightUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp*hammer_bglvar')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp*hammer_bglvar*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp')

                shapes[sname + '_puWeightDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown*hammer_bglvar')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown*hammer_bglvar*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown')
        
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
            shapes['jpsi_mu_'+new_name] = shapes['jpsi_mu_'+new_name].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*'+ham)
            shapes['jpsi_tau_'+new_name] = samples['jpsi_tau']
            shapes['jpsi_tau_'+new_name] = shapes['jpsi_tau_'+new_name].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*'+ham+'*%f*%f' %(blind,rjpsi))

        if flat_fakerate == False:
            for name in shapes:
                shapes[name] = shapes[name].Define('shape_weight_wfr','shape_weight*nn/(1-nn)')

    colours = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][len(samples)]))

    # CREATE THE SMART POINTERS IN ONE GO AND PRODUCE RESULTS IN ONE SHOT,
    # SEE MAX GALLI PRESENTATION
    # https://github.com/maxgalli/dask-pyroot-tutorial/blob/master/2_rdf_basics.ipynb
    # https://indico.cern.ch/event/882824/contributions/3929999/attachments/2073718/3481850/PyROOT_PyHEP_2020.pdf

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
            if k not in datacards:
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
        c1.cd()
        leg = create_legend(temp_hists, sample_names, titles)
        main_pad.cd()
        main_pad.SetLogy(False)
        
        maxima = [] # save maxima for the look of the stack plot
        data_max = 0.
        for i, kv in enumerate(temp_hists[k].items()):
            key = kv[0]
            ihist = kv[1]
            ihist.GetXaxis().SetTitle(v[1])
            ihist.GetYaxis().SetTitle('events')
            ihist.SetLineColor(colours[i] if key!='%s_data'%k else ROOT.kBlack)
            ihist.SetFillColor(colours[i] if key!='%s_data'%k else ROOT.kWhite)
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
        fakes.SetFillColor(colours[len(samples)-1])
        fakes.SetFillStyle(1001)
        fakes.SetLineColor(colours[len(samples)-1])
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
    
        temp_hists[k]['%s_data'%k].Draw('EP SAME')
        CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
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
        ratio.Draw('EP same')
    
        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_ul/%s/pdf/lin/%s.pdf' %(label, k))
        c1.SaveAs('plots_ul/%s/png/lin/%s.png' %(label, k))
    
        ths1.SetMaximum(20*max(sum(maxima), data_max))
        ths1.SetMinimum(10)
        main_pad.SetLogy(True)
        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_ul/%s/pdf/log/%s.pdf' %(label, k))
        c1.SaveAs('plots_ul/%s/png/log/%s.png' %(label, k))
        
        if k in datacards and shape_nuisances:
            create_datacard_prep(temp_hists[k],unc_hists[k],shapes,'pass',k,label)
            plot_shape_nuisances(label, k, 'pass')
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
        ratio_fake.Draw('EP same')

        c1.Modified()
        c1.Update()
        if flat_fakerate:
            c1.SaveAs('plots_ul/%s/fail_region/pdf/lin/%s.pdf' %(label, k))
            c1.SaveAs('plots_ul/%s/fail_region/png/lin/%s.png' %(label, k))
        else:
            c1.SaveAs('plots_ul/%s/fail_region_reweight/pdf/lin/%s.pdf' %(label, k))
            c1.SaveAs('plots_ul/%s/fail_region_reweight/png/lin/%s.png' %(label, k))

        ths1_fake.SetMaximum(20*max(sum(maxima), data_max))
        ths1_fake.SetMinimum(10)
        main_pad.SetLogy(True)
        c1.Modified()
        c1.Update()
        if flat_fakerate:
            c1.SaveAs('plots_ul/%s/fail_region/pdf/log/%s.pdf' %(label, k))
            c1.SaveAs('plots_ul/%s/fail_region/png/log/%s.png' %(label, k))
        else:
            c1.SaveAs('plots_ul/%s/fail_region_reweight/pdf/log/%s.pdf' %(label, k))
            c1.SaveAs('plots_ul/%s/fail_region_reweight/png/log/%s.png' %(label, k))

        if k in datacards and shape_nuisances:
            create_datacard_prep(temp_hists_fake[k],unc_hists_fake[k],shapes,'fail',k,label)
            plot_shape_nuisances(label, k, 'fail')
        
    save_yields(label, temp_hists)
    save_selection(label, preselection)
    save_weights(label, sample_names, weights)


# save reduced trees to produce datacards
# columns = ROOT.std.vector('string')()
# for ic in ['Q_sq', 'm_miss_sq', 'E_mu_star', 'E_mu_canc', 'Bmass']:
#     columns.push_back(ic)
# for k, v in samples.items():
#     v.Snapshot('tree', 'plots_ul/%s/tree_%s_datacard.root' %(label, k), columns)
