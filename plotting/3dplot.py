import ROOT
import os
import time
from datetime import datetime
from bokeh.palettes import viridis, all_palettes
import random
import math

#cms libraries
from officialStyle import officialStyle
from cmsstyle import CMS_lumi

# personal libraries
from samples import weights, sample_names, titles, colours
from new_branches import to_define
from selections import preselection, preselection_mc, pass_id, fail_id
from create_datacard_3dfit import create_datacard_pass,create_datacard_fail
from plot_shape_nuisances import plot_shape_nuisances

#options
shape_nuisances = True
flat_fakerate = False # false mean that we use the NN weights for the fr
blind_analysis = True
rjpsi = 1

start_time = time.time()

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

def make_directories(label):
    '''
    Function to create directories to save the plots
    '''
    if not os.path.exists('multi_plots'): os.system('mkdir -p multi_plots')
    # For the 2D plots
    os.system('mkdir -p multi_plots/%s/lego/pdf/' %label)
    os.system('mkdir -p multi_plots/%s/lego/png/' %label)
    os.system('mkdir -p multi_plots/%s/colz/pdf/' %label)
    os.system('mkdir -p multi_plots/%s/colz/png/' %label)
    os.system('mkdir -p multi_plots/%s/unrolled/pdf/' %label)
    os.system('mkdir -p multi_plots/%s/unrolled/png/' %label)
    os.system('mkdir -p multi_plots/%s/unrolled/fail_region/pdf/' %label)
    os.system('mkdir -p multi_plots/%s/unrolled/fail_region/png/' %label)
    os.system('mkdir -p multi_plots/%s/datacards' %label)

def create_legend(temp_hists, sample_names, titles):
    '''
    Function to create the legend for the plots
    '''
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


def create_datacard_prep(hists, shape_hists, shapes_names, pf_flag, q2_region, label, nchannel):
    '''
    Creates and saves the root file with the histograms of each contribution.
    Saves the histograms of the shape nuisances.
    Calls the 'create datacard' function, both for the pass and fail regions, 
    to write the text datacard for the fit in combine. 
    '''
    fout = ROOT.TFile.Open('multi_plots/%s/datacards/datacard_%s_%s.root' %(label, pf_flag, q2_region), 'UPDATE')
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
    fout.Close()
    # datacard txt are different depending on the region
    if pf_flag == 'pass':
        create_datacard_pass(label, q2_region, nchannel)
    else:
        create_datacard_fail(label, q2_region, nchannel+4)


def make_binbybin(hist, flag, label, name):
    fout = ROOT.TFile.Open('multi_plots/%s/datacards/datacard_%s_%s.root' %(label,flag, name), 'RECREATE')
    for i in range(1,hist.GetNbinsX()+1):
        histo_up = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+"_"+name+"_"+flag+'Up','',hist.GetNbinsX(),hist.GetBinLowEdge(1), hist.GetBinLowEdge(hist.GetNbinsX() + 1))
        histo_down = ROOT.TH1D('jpsi_x_mu_bbb'+str(i)+"_"+name+"_"+flag+'Down','',hist.GetNbinsX(),hist.GetBinLowEdge(1), hist.GetBinLowEdge(hist.GetNbinsX() + 1))
        for nbin in range(1,hist.GetNbinsX()+1):
            if nbin == i:
                histo_up.SetBinContent(nbin,hist.GetBinContent(nbin) + hist.GetBinError(nbin))
                histo_up.SetBinError(nbin,hist.GetBinError(nbin) + math.sqrt(hist.GetBinError(nbin)))
                histo_down.SetBinContent(nbin,hist.GetBinContent(nbin) - hist.GetBinError(nbin))
                histo_down.SetBinError(nbin,hist.GetBinError(nbin) - math.sqrt(hist.GetBinError(nbin)))
            else:
                histo_up.SetBinContent(nbin,hist.GetBinContent(nbin))
                histo_up.SetBinError(nbin,hist.GetBinError(nbin))
                histo_down.SetBinContent(nbin,hist.GetBinContent(nbin))
                histo_down.SetBinError(nbin,hist.GetBinError(nbin))
        fout.cd()
        histo_up.Write()
        histo_down.Write()
    fout.Close()

def save_weights(label, sample_names, weights):
    with open('multi_plots/%s/normalisations.txt' %label, 'w') as ff:
        for sname in sample_names: 
            print(sname+'\t\t%.2f' %weights[sname], file=ff)
        print("Flat fake rate weight %s" %str(flat_fakerate), file = ff)

def save_selection(label, preselection):
    with open('multi_plots/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '): 
            print("    '%s'," %isel, file=ff)
        print('])', file=ff)
        print('pass: %s'%pass_id, file=ff)
        print('fail: %s'%fail_id, file=ff)


if __name__ == '__main__':
    
    c1 = ROOT.TCanvas('c1', '', 700, 700)
    c1.Draw()
    c1.cd()
    main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
    main_pad.Draw()

    # timestamp
    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    # create plot directories
    make_directories(label)

    # Input flat ntuples
    samples = dict()
    tree_name = 'BTo3Mu'
    tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021May31_nn'

    for k in sample_names:
        if k == 'data':
            samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_fakerate.root'%(tree_dir,k)) 
        else:
            samples[k] = ROOT.RDataFrame(tree_name,'%s/%s_sf.root'%(tree_dir,k))

    print("======> Samples loaded")

    #Blind analysis: hide the value of rjpsi for the fit
    if blind_analysis:
        random.seed(2)
        rand = random.randint(0, 10000)
        blind = rand/10000 *1.5 +0.5
    else:
        blind = 1.
    
    #################################################
    ####### Weights ################################
    #################################################
    for k, v in samples.items():
        samples[k] = samples[k].Define('br_weight', '%f' %weights[k])
        #for jpsi tau apply ctau, pu and ff weights. Plus the values for the blind analyss and rjpsi
        if k=='jpsi_tau':
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_total*%f*%f' %(blind,rjpsi))

        # jpsi mu apply ctau, pu and ff weights
        elif k=='jpsi_mu':
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_total')

        #For all the other samples we apply ctau and pu
        #For the Bc samples the ctau contribution is != 1., while for the background it is ==1
        else:
            samples[k] = samples[k].Define('total_weight', 'ctau_weight_central*br_weight*puWeight*sf_total' if k!='data' else 'br_weight') 
        
        #define new columns
        for new_column, new_definition in to_define: 
            if samples[k].HasColumn(new_column):
                continue
            samples[k] = samples[k].Define(new_column, new_definition)

    # If the fakerate is not flat, we compute the weights with the nn weight
    if flat_fakerate == False:
        for sample in samples:
            samples[sample] = samples[sample].Define('total_weight_wfr', 'total_weight*nn/(1-nn)')

    ###########################
    #### Apply preselection ###
    ###########################
    for k, v in samples.items():
        filter = preselection_mc if k!='data' else preselection
        samples[k] = samples[k].Filter(filter)


    #################################################
    ####### Shape nuisances definition ##############
    #################################################

    # Create a new dictionary "shapes", similar to the "samples" one defined for the datasets
    # Each entry of the dic is a nuisance for a different dataset
    if shape_nuisances :
        shapes = dict()
        #ctau nuisances
        for sname in samples:
            #Only Bc samples want this nuisance
            if (sname != 'jpsi_x_mu' and sname != 'data' ):
                shapes[sname + '_ctauUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight*hammer_bglvar*sf_total')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight*hammer_bglvar*sf_total*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_ctauUp'] = shapes[sname + '_ctauUp'].Define('shape_weight', 'ctau_weight_up*br_weight*puWeight*sf_total')
                shapes[sname + '_ctauDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight*hammer_bglvar*sf_total')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight*hammer_bglvar*sf_total*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_ctauDown'] = shapes[sname + '_ctauDown'].Define('shape_weight', 'ctau_weight_down*br_weight*puWeight*sf_total')
        
            # Pile up nuisances
            if (sname != 'data'):
                shapes[sname + '_puWeightUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp*hammer_bglvar*sf_total')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp*hammer_bglvar*sf_total*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_puWeightUp'] = shapes[sname + '_puWeightUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightUp*sf_total')

                shapes[sname + '_puWeightDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown*hammer_bglvar*sf_total')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown*hammer_bglvar*sf_total*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_puWeightDown'] = shapes[sname + '_puWeightDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeightDown*sf_total')
                
            #scale factor reco
            if (sname != 'data'):
                shapes[sname + '_sfRecoUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_sfRecoUp'] = shapes[sname + '_sfRecoUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_reco_up')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_sfRecoUp'] = shapes[sname + '_sfRecoUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_reco_up*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_sfRecoUp'] = shapes[sname + '_sfRecoUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_reco_up')

                shapes[sname + '_sfRecoDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_sfRecoDown'] = shapes[sname + '_sfRecoDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_reco_down')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_sfRecoDown'] = shapes[sname + '_sfRecoDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_reco_down*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_sfRecoDown'] = shapes[sname + '_sfRecoDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_reco_down')

            #scale factor id
            if (sname != 'data'):
                shapes[sname + '_sfIdUp'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname +'_sfIdUp'] = shapes[sname + '_sfIdUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_id_up')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_sfIdUp'] = shapes[sname + '_sfIdUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_id_up*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_sfIdUp'] = shapes[sname + '_sfIdUp'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_id_up')

                shapes[sname + '_sfIdDown'] = samples[sname]
                if sname == 'jpsi_mu':
                    shapes[sname + '_sfIdDown'] = shapes[sname + '_sfIdDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_id_down')
                elif sname == 'jpsi_tau':
                    shapes[sname +'_sfIdDown'] = shapes[sname + '_sfIdDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*hammer_bglvar*sf_id_down*%f*%f' %(blind,rjpsi))
                else:
                    shapes[sname + '_sfIdDown'] = shapes[sname + '_sfIdDown'].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_id_down')


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
            shapes['jpsi_mu_'+new_name] = shapes['jpsi_mu_'+new_name].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_total*'+ham)
            shapes['jpsi_tau_'+new_name] = samples['jpsi_tau']
            shapes['jpsi_tau_'+new_name] = shapes['jpsi_tau_'+new_name].Define('shape_weight', 'ctau_weight_central*br_weight*puWeight*sf_total*'+ham+'*%f*%f' %(blind,rjpsi))

        if flat_fakerate == False:
            for name in shapes:
                shapes[name] = shapes[name].Define('shape_weight_wfr','shape_weight*nn/(1-nn)')

    ##############################################################################


    ####################################################
    ###############   Q2 REGIONS #######################
    ####################################################

    # 2D plots in 4 different regions of q^2
    q2_bins = {
               'aa':'Q_sq>=0 && Q_sq<6',
               'bb':'Q_sq>=6 && Q_sq<8',
               'cc':'Q_sq>=8 && Q_sq<8.7',
               'dd':'Q_sq>=8.7 && Q_sq<=10'
    }

    #define 2D histos in the 4 qsq regions for each sample (e_mu_star vs m_miss_sq)
    histos = {}
    nbins = 25
    for iter_q2,k in enumerate(q2_bins):
        histos['estar_mmiss_'+k] = ROOT.RDF.TH2DModel('estar_mmiss_'+k, '', nbins, 0.3, 2.3, nbins, 0, 9)

    temp_hists           = {}
    temp_hists_fake      = {}
    for iter_q2,k in enumerate(q2_bins):
        temp_hists[k]           = {}
        temp_hists_fake[k]      = {}
        for kk, vv in samples.items():
            temp_hists[k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([pass_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','total_weight')
            if flat_fakerate:
                temp_hists_fake[k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([fail_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','total_weight')
            else:
                temp_hists_fake[k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([fail_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','total_weight_wfr')
        
    # Create pointers for the shapes histos 
    if shape_nuisances:
        print('====> shape uncertainties histos')
        unc_hists      = {} # pass muon ID category
        unc_hists_fake = {} # pass muon ID category
        for k in q2_bins:
            # Compute them only for the variables that we want to fit
            unc_hists     [k] = {}
            unc_hists_fake[k] = {}
            for kk, vv in shapes.items():
                unc_hists      [k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([pass_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','shape_weight')
                if flat_fakerate:
                    unc_hists_fake[k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([fail_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','shape_weight')
                else:
                    unc_hists_fake[k]['%s_%s'%(k,kk)]= vv.Filter(' && '.join([fail_id,q2_bins[k]])).Histo2D(histos['estar_mmiss_'+k],'E_mu_star','m_miss_sq','shape_weight_wfr')

    ######################################
    ############## PLOTS #################
    ######################################

    print('====> now looping on q2 regions')
    #graphical features
    # each q2 region is a different plot
    for iter_q2,k in enumerate(q2_bins):
        c1.cd()
        leg = create_legend(temp_hists, sample_names, titles)
        main_pad.cd()
        main_pad.SetLogy(False)

        maxima = [] # save maxima for the look of the stack plot
        data_max = 0.        
        #Graphical features for the 2D histos (needed?)
        for i, kv in enumerate(temp_hists[k].items()):
            key = kv[0]
            ihist = kv[1]
            sample_name = key.split(k+'_')[1]
            ihist.GetXaxis().SetTitle("E_{#mu}*")
            ihist.GetYaxis().SetTitle('m_{miss}^2')
            ihist.SetLineColor(colours[sample_name])
            ihist.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
            if key!='%s_data'%k:
                maxima.append(ihist.GetMaximum())
            else:
                data_max = ihist.GetMaximum()

	
        # Definition of stack histos for 2D plots
        ths1      = ROOT.THStack('stack', '')
        ths1_fake = ROOT.THStack('stack_fake', '')

        # stack histos for the unrolled histos (the 2D histo is unrolled to be used with combine)
        ths1_unrolled      = ROOT.THStack('unrolled stack', '')
        ths1_unrolled_fake = ROOT.THStack('unrolled stack fake', '')
        
        for i, kv in enumerate(temp_hists[k].items()):
            key = kv[0]
            ihist = kv[1]            
            # adding 2d histos to their stack
            if key=='%s_data'%k: continue
            #ihist.Draw('hist' + 'same'*(i>0))
            ths1.Add(ihist.GetValue())

        # apply same aestethics to pass and fail (needed?)
        for kk in temp_hists[k].keys():
            temp_hists_fake[k][kk].GetXaxis().SetTitle(temp_hists[k][kk].GetXaxis().GetTitle())
            temp_hists_fake[k][kk].GetYaxis().SetTitle(temp_hists[k][kk].GetYaxis().GetTitle())
            temp_hists_fake[k][kk].SetLineColor(temp_hists[k][kk].GetLineColor())
            temp_hists_fake[k][kk].SetFillColor(temp_hists[k][kk].GetFillColor())

        # Fakes controbution from data in the fail region
        temp_hists[k]['%s_fakes' %k] = temp_hists_fake[k]['%s_data' %k].Clone()
        fakes = temp_hists[k]['%s_fakes' %k]
        # Subtract to fakes all the contributions of other samples in the fail region
        for i, kv in enumerate(temp_hists_fake[k].items()):
            if 'data' in kv[0]:
                kv[1].SetLineColor(ROOT.kBlack)
                continue
            else:
                fakes.Add(kv[1].GetPtr(), -1.)

        fakes.SetFillColor(colours['fakes'])
        fakes.SetFillStyle(1001)
        fakes.SetLineColor(colours['fakes'])
        fakes_forfail = fakes.Clone()
        if flat_fakerate:
            fakes.Scale(weights['fakes'])
        ths1.Add(fakes)


        #####################################
        ########### UNROLLED HISTOS #########
        #####################################

        #dic of final unrolled_bins
        hists_unrolled = {}
        hists_unrolled_fake = {}

        #dic of tmp unrolled_bins (with empty bins also)
        tmp_hists_unrolled = {}
        tmp_hists_unrolled_fake = {}

        #####################################
        ########### Pass Region ### #########
        #####################################

        # Compute the unrolled stack histo (with empty bins also)
        maxima_unrolled = []
        for i, kv in enumerate(temp_hists[k].items()):
            key = kv[0]
            ihist = kv[1]
            tmp_ihist_unrolled = ROOT.TH1F('unrolled '+ key,'unrolled '+ key,ihist.GetNbinsY()*ihist.GetNbinsX(),0.,ihist.GetNbinsY()*ihist.GetNbinsX())
            for ybin in range(1,ihist.GetNbinsY()):
                for xbin in range(1,ihist.GetNbinsX()):
                    tmp_ihist_unrolled.SetBinContent(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinContent(xbin,ybin))
                    tmp_ihist_unrolled.SetBinError(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinError(xbin,ybin))
            sample_name = key.split(k+'_')[1]
            tmp_ihist_unrolled.SetLineColor(colours[sample_name])
            tmp_ihist_unrolled.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
            #tmp_ihist_unrolled.Draw('hist'+'same'*(i>0))
            tmp_hists_unrolled[key] = tmp_ihist_unrolled
            maxima_unrolled.append(tmp_ihist_unrolled.GetMaximum())


        # Compute the unrolled histo for the fakes sample, to add to the total dictionary of unrolled histos
        tmp_fakes_unrolled = ROOT.TH1F('unrolled '+ key,'unrolled '+ key,ihist.GetNbinsY()*ihist.GetNbinsX(),0.,ihist.GetNbinsY()*ihist.GetNbinsX())
        for ybin in range(1,fakes.GetNbinsY()):
            for xbin in range(1,fakes.GetNbinsX()):
                tmp_fakes_unrolled.SetBinContent(xbin + fakes.GetNbinsX()*(ybin -1), fakes.GetBinContent(xbin,ybin))
                tmp_fakes_unrolled.SetBinError(xbin + fakes.GetNbinsX()*(ybin -1), fakes.GetBinError(xbin,ybin))
        tmp_hists_unrolled['%s_fakes'%k] = tmp_fakes_unrolled

        # Create unrolled_histo that do not have empty bins
        # in chosen_bins, the final choice of bins that are not empty
        total_bins = [i for i in range(1,tmp_hists_unrolled['%s_data'%k].GetNbinsX()+1)]
        chosen_bins = []
        # loop over the total bins
        for b in range(1,len(total_bins)+1):
            #loop over the samples
            flag_not_empty = 0. # this flag becomes 1. if the bin is not empty for at least one sample 
            for i, kv in enumerate(temp_hists[k].items()):
                key = kv[0]
                #check which bins to keep
                if tmp_hists_unrolled[key].GetBinContent(b) >= 80.:
                    flag_not_empty = 1.
                else:
                    if key == '%s_data'%k: #if data is empty, we skip the bin anyway
                        flag_not_empty = 0.
                        break 
            if flag_not_empty:
                chosen_bins.append(b)
        
        # create the non empty histos + stack
        for i, kv in enumerate(temp_hists[k].items()):
            key = kv[0]
            ihist_unrolled = ROOT.TH1F('clean_unrolled_'+ key,'', len(chosen_bins),0., len(chosen_bins))
            ibin = 1
            for b in range(1,tmp_hists_unrolled[key].GetNbinsX()+1):
                if b in chosen_bins:
                    if tmp_hists_unrolled[key].GetBinContent(b) <= 0.:
                        ihist_unrolled.SetBinContent(ibin, 0.01)
                    else:
                        ihist_unrolled.SetBinContent(ibin,tmp_hists_unrolled[key].GetBinContent(b))
                    ihist_unrolled.SetBinError(ibin,tmp_hists_unrolled[key].GetBinError(b))
                    ibin += 1
            sample_name = key.split(k+'_')[1]
            ihist_unrolled.SetLineColor(colours[sample_name])
            ihist_unrolled.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
            #ihist_unrolled.Draw('hist'+'same'*(i>0))
            hists_unrolled[key]=ihist_unrolled
            if key == '%s_data'%k: continue # we don't want data in the stack
            ths1_unrolled.Add(ihist_unrolled)

        #####################################
        ########### Fail Region #############
        #####################################

        maxima_unrolled_fake = []
        for i, kv in enumerate(temp_hists_fake[k].items()):
            key = kv[0]
            ihist = kv[1]
            # Compute the unrolled stack histo (with empty bins also)
            tmp_ihist_unrolled = ROOT.TH1F('unrolled '+ key,'unrolled '+ key,ihist.GetNbinsY()*ihist.GetNbinsX(),0.,ihist.GetNbinsY()*ihist.GetNbinsX())
            for ybin in range(1,ihist.GetNbinsY()):
                for xbin in range(1,ihist.GetNbinsX()):
                    tmp_ihist_unrolled.SetBinContent(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinContent(xbin,ybin))
                    tmp_ihist_unrolled.SetBinError(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinError(xbin,ybin))
            sample_name = key.split(k+'_')[1]
            tmp_ihist_unrolled.SetLineColor(colours[sample_name])
            tmp_ihist_unrolled.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
            #tmp_ihist_unrolled.Draw('hist'+'same'*(i>0))
            tmp_hists_unrolled_fake[key] = tmp_ihist_unrolled
            maxima_unrolled_fake.append(tmp_ihist_unrolled.GetMaximum())

        tmp_hists_unrolled_fake['%s_fakes'%k] = tmp_fakes_unrolled

        '''# in chosen_bins, the final choice of bins that are not empty
        total_bins_fake = [i for i in range(1,tmp_hists_unrolled_fake['%s_data'%k].GetNbinsX()+1)]
        chosen_bins_fake = []
        # loop over the total bins
        for b in range(1,len(total_bins_fake)+1):
            #loop over the samples
            flag_not_empty = 0. # this flag becomes 1. if the bin is not empty for at least one sample 
            for i, kv in enumerate(tmp_hists_unrolled_fake.keys()):
                key = kv
                if tmp_hists_unrolled_fake[key].GetBinContent(b) >= 0.0001:
                    flag_not_empty = 1.
            if flag_not_empty:
                chosen_bins_fake.append(b)
        '''
        # create the non empty histos + stack
        for i, kv in enumerate(tmp_hists_unrolled_fake.keys()):
            key = kv
            ihist_unrolled = ROOT.TH1F('clean_unrolled_'+ key,'', len(chosen_bins),0., len(chosen_bins))
            ibin = 1
            for b in range(1,tmp_hists_unrolled_fake[key].GetNbinsX()+1):
                if b in chosen_bins:
                    if tmp_hists_unrolled_fake[key].GetBinContent(b) <= 0.:
                        ihist_unrolled.SetBinContent(ibin, 0.01)
                    else:
                        ihist_unrolled.SetBinContent(ibin,tmp_hists_unrolled_fake[key].GetBinContent(b))
                    ihist_unrolled.SetBinError(ibin,tmp_hists_unrolled_fake[key].GetBinError(b))
                    ibin += 1
            sample_name = key.split(k+'_')[1]
            ihist_unrolled.SetLineColor(colours[sample_name])
            ihist_unrolled.SetFillColor(colours[sample_name] if key!='%s_data'%k else ROOT.kWhite)
            #ihist_unrolled.Draw('hist'+'same'*(i>0))
            hists_unrolled_fake[key]=ihist_unrolled
            if key == '%s_data'%k: continue # we don't want data in the stack
            ths1_unrolled_fake.Add(ihist_unrolled)
        
        #############################################################
        ########### Unrolled histos for shape nuisances #############
        ############################################################
        if shape_nuisances:
            #dic of final unrolled_bins
            shapes_hists_unrolled = {}
            shapes_hists_unrolled_fake = {}
            
            #dic of tmp unrolled_bins (with empty bins also)
            tmp_shapes_hists_unrolled = {}
            tmp_shapes_hists_unrolled_fake = {}
            
            
            #####################################
            ########### Pass Region ### #########
            #####################################
            
            #loop on all the shapes
            for i, kv in enumerate(unc_hists[k].items()):
                key = kv[0]
                ihist = kv[1]
                unc_ihist_unrolled = ROOT.TH1F('unrolled '+ key,'unrolled '+ key,ihist.GetNbinsY()*ihist.GetNbinsX(),0.,ihist.GetNbinsY()*ihist.GetNbinsX())
                for ybin in range(1,ihist.GetNbinsY()):
                    for xbin in range(1,ihist.GetNbinsX()):
                        unc_ihist_unrolled.SetBinContent(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinContent(xbin,ybin))
                        unc_ihist_unrolled.SetBinError(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinError(xbin,ybin))
                sample_name = key.split(k+'_')[1]
                tmp_shapes_hists_unrolled[key] = unc_ihist_unrolled
	
            #chosen bins is the same as before	        
            # create the non empty histos + stack
            for i, kv in enumerate(unc_hists[k].items()):
                key = kv[0]
                ihist_unrolled = ROOT.TH1F('clean_unrolled_'+ key,'', len(chosen_bins),0., len(chosen_bins))
                ibin = 1
                for b in range(1,tmp_shapes_hists_unrolled[key].GetNbinsX()+1):
                    if b in chosen_bins:
                        if tmp_shapes_hists_unrolled[key].GetBinContent(b) <= 0.:
                            ihist_unrolled.SetBinContent(ibin, 0.01)
                        else:
                            ihist_unrolled.SetBinContent(ibin,tmp_shapes_hists_unrolled[key].GetBinContent(b))
                        ihist_unrolled.SetBinError(ibin,tmp_shapes_hists_unrolled[key].GetBinError(b))
                        ibin += 1
                sample_name = key.split(k+'_')[1]
                shapes_hists_unrolled[key]=ihist_unrolled

            #####################################
            ########### Fail Region #############
            #####################################
            
            for i, kv in enumerate(unc_hists_fake[k].items()):
                key = kv[0]
                ihist = kv[1]
                # Compute the unrolled stack histo (with empty bins also)
                unc_ihist_unrolled = ROOT.TH1F('unrolled '+ key,'unrolled '+ key,ihist.GetNbinsY()*ihist.GetNbinsX(),0.,ihist.GetNbinsY()*ihist.GetNbinsX())
                for ybin in range(1,ihist.GetNbinsY()):
                    for xbin in range(1,ihist.GetNbinsX()):
                        unc_ihist_unrolled.SetBinContent(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinContent(xbin,ybin))
                        unc_ihist_unrolled.SetBinError(xbin + ihist.GetNbinsX()*(ybin -1), ihist.GetBinError(xbin,ybin))
                tmp_shapes_hists_unrolled_fake[key] = unc_ihist_unrolled
                	
            	        
            # create the non empty histos + stack
            for i, kv in enumerate(tmp_shapes_hists_unrolled_fake.keys()):
                key = kv
                ihist_unrolled = ROOT.TH1F('clean_unrolled_'+ key,'', len(chosen_bins),0., len(chosen_bins))
                ibin = 1
                for b in range(1,tmp_shapes_hists_unrolled_fake[key].GetNbinsX()+1):
                    if b in chosen_bins:
                        if tmp_shapes_hists_unrolled_fake[key].GetBinContent(b) <= 0.:
                            ihist_unrolled.SetBinContent(ibin, 0.01)
                        else:
                            ihist_unrolled.SetBinContent(ibin,tmp_shapes_hists_unrolled_fake[key].GetBinContent(b))
                        ihist_unrolled.SetBinError(ibin,tmp_shapes_hists_unrolled_fake[key].GetBinError(b))
                        ibin += 1
                shapes_hists_unrolled_fake[key]=ihist_unrolled

        ######################################################################
        if shape_nuisances:
            make_binbybin(hists_unrolled['%s_jpsi_x_mu'%k],'pass', label, k)
            make_binbybin(hists_unrolled_fake['%s_jpsi_x_mu'%k],'fail', label, k)

            create_datacard_prep(hists = hists_unrolled, shape_hists = shapes_hists_unrolled, shapes_names= shapes, pf_flag = 'pass', q2_region = k, label = label, nchannel = iter_q2+1)
            plot_shape_nuisances(label, k, 'pass', path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/multi_plots/', plot3d = True)
            
            create_datacard_prep(hists = hists_unrolled_fake, shape_hists = shapes_hists_unrolled_fake, shapes_names= shapes, pf_flag = 'fail', q2_region = k, label = label, nchannel = iter_q2+1)
            plot_shape_nuisances(label, k, 'fail', path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/multi_plots/', plot3d = True)

        
        ###############################
        ######## Draw Lego plot #######
        ###############################

        ths1.Draw('lego1 0')
        ths1.GetXaxis().SetTitle('E_{#mu}*')
        ths1.GetYaxis().SetTitle('m_{miss}^{2}')
        ths1.SetMinimum(0.)

        if flat_fakerate:
            leg.AddEntry(fakes, 'fakes flat', 'F')    
        else:
            leg.AddEntry(fakes, 'fakes nn', 'F')    

        temp_hists[k]['%s_data'%k].Draw('EP SAME')
        CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
        main_pad.cd()

        c1.Modified()
        c1.Update()
        c1.SaveAs('multi_plots/%s/lego/pdf/%s.pdf' %(label, k))
        c1.SaveAs('multi_plots/%s/lego/png/%s.png' %(label, k))

        ###############################
        ######## Draw COLZ plot #######
        ###############################

        
        ths1.Draw('COLZ')
        c1.Modified()
        c1.Update()
        c1.SaveAs('multi_plots/%s/colz/pdf/%s.pdf' %(label, k))
        c1.SaveAs('multi_plots/%s/colz/png/%s.png' %(label, k))

        ########################################################
        ############# Draw Unrolled histos #####################
        ########################################################

        #redefine c2 bc we want also the ratio plot here
        c2 = ROOT.TCanvas('c2', '', 700, 700)
        c2.Draw()
        c2.cd()
        main_pad2 = ROOT.TPad('main_pad2', '', 0., 0.25, 1. , 1.  )
        main_pad2.Draw()
        c2.cd()
        ratio_pad2 = ROOT.TPad('ratio_pad2', '', 0., 0., 1., 0.25)
        ratio_pad2.Draw()

        main_pad2.SetTicks(True)
        main_pad2.SetBottomMargin(0.)

        ratio_pad2.SetTopMargin(0.)   
        ratio_pad2.SetGridy()
        ratio_pad2.SetBottomMargin(0.45)

        main_pad2.cd()
        main_pad2.SetLogy(False)

        ths1_unrolled.Draw("hist")
        ths1_unrolled.GetXaxis().SetTitle('Unrolled 2D bins')
        ths1_unrolled.GetYaxis().SetTitle('Events')
        ths1_unrolled.SetMaximum(max(maxima_unrolled) * 1.7)
        ths1_unrolled.SetMinimum(0.)

        # statistical uncertainty
        stats = ths1_unrolled.GetStack().Last().Clone()
        stats.SetLineColor(0)
        stats.SetFillColor(ROOT.kGray+1)
        stats.SetFillStyle(3344)
        stats.SetMarkerSize(0)
        stats.Draw('E2 SAME')

        leg.AddEntry(stats, 'stat. unc.', 'F')
        leg.Draw('same')
        hists_unrolled['%s_data'%k].Draw('EP SAME')

        CMS_lumi(main_pad2, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
        main_pad2.cd()

        # if the analisis if blind, we don't want to show the rjpsi prefit value
        if not blind_analysis:
            rjpsi_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
            rjpsi_value.AddText('R(J/#Psi) = %.2f' %rjpsi)
            rjpsi_value.SetFillColor(0)
            rjpsi_value.Draw('EP')

        q2_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
        q2_value.AddText(q2_bins[k])
        q2_value.SetFillColor(0)
        q2_value.Draw('EP')

        # Ratio for pass region
        ratio_pad2.cd()
        ratio = hists_unrolled['%s_data'%k].Clone()
        ratio.SetName(ratio.GetName()+'_ratio')
        ratio.Divide(stats)
        ratio_stats = stats.Clone()
        ratio_stats.SetName(ratio.GetName()+'_ratiostats')
        ratio_stats.Divide(stats)
        ratio_stats.SetMaximum(1.999) # avoid displaying 2, that overlaps with 0 in the main_pad2
        ratio_stats.SetMinimum(0.001) # and this is for symmetry
        ratio_stats.GetYaxis().SetTitle('obs/exp')
        ratio_stats.GetYaxis().SetTitleOffset(0.5)
        ratio_stats.GetXaxis().SetTitle('Unrolled 2D bins')
        ratio_stats.GetXaxis().SetTitleOffset(0.5)
        ratio_stats.GetYaxis().SetNdivisions(405)
        ratio_stats.GetXaxis().SetLabelSize(3.* ratio.GetXaxis().GetLabelSize())
        ratio_stats.GetYaxis().SetLabelSize(3.* ratio.GetYaxis().GetLabelSize())
        ratio_stats.GetXaxis().SetTitleSize(3.* ratio.GetXaxis().GetTitleSize())
        ratio_stats.GetYaxis().SetTitleSize(3.* ratio.GetYaxis().GetTitleSize())

        norm_stack = ROOT.THStack('norm_stack', '')

        for key in hists_unrolled.keys():
            if key == '%s_data'%k: continue
            vv = hists_unrolled[key]
            hh = vv.Clone()
            hh.Divide(stats)
            norm_stack.Add(hh)

        line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(1)
        ratio_stats.Draw('E2')
        norm_stack.Draw('hist same')
        ratio_stats.Draw('E2 same')
        line.Draw('same')
        ratio.Draw('EP same')
    
        c2.Modified()
        c2.Update()
        c2.SaveAs('multi_plots/%s/unrolled/pdf/%s.pdf' %(label, k))
        c2.SaveAs('multi_plots/%s/unrolled/png/%s.png' %(label, k))



        ###################################################################
        # Now creating and saving the stack of the FAIL REGION
        c2.cd()
        main_pad2.cd()
        main_pad2.SetLogy(False)
        ths1_unrolled_fake.Draw("hist")
        ths1_unrolled_fake.GetXaxis().SetTitle('Unrolled 2D bins')
        ths1_unrolled_fake.GetYaxis().SetTitle('Events')
        ths1_unrolled_fake.SetMaximum(max(maxima_unrolled_fake) * 1.7)
        ths1_unrolled_fake.SetMinimum(0.)

        # statistical uncertainty
        stats_fake = ths1_unrolled_fake.GetStack().Last().Clone()
        stats_fake.SetLineColor(0)
        stats_fake.SetFillColor(ROOT.kGray+1)
        stats_fake.SetFillStyle(3344)
        stats_fake.SetMarkerSize(0)
        stats_fake.Draw('E2 SAME')

        leg.AddEntry(stats_fake, 'stat. unc.', 'F')
        leg.Draw('same')
        hists_unrolled_fake['%s_data'%k].Draw('EP SAME')

        CMS_lumi(main_pad2, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
        main_pad2.cd()

        q2_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
        q2_value.AddText(q2_bins[k])
        q2_value.SetFillColor(0)
        q2_value.Draw('EP')

        # Ratio for pass region
        ratio_pad2.cd()
        ratio_fake = hists_unrolled_fake['%s_data'%k].Clone()
        ratio_fake.SetName(ratio_fake.GetName()+'_ratio_fake')
        ratio_fake.Divide(stats_fake)
        ratio_stats_fake = stats_fake.Clone()
        ratio_stats_fake.SetName(ratio_fake.GetName()+'_ratiostats_fake')
        ratio_stats_fake.Divide(stats_fake)
        ratio_stats_fake.SetMaximum(1.999) # avoid displaying 2, that overlaps with 0 in the main_pad2
        ratio_stats_fake.SetMinimum(0.001) # and this is for symmetry
        ratio_stats_fake.GetYaxis().SetTitle('obs/exp')
        ratio_stats_fake.GetYaxis().SetTitleOffset(0.5)
        ratio_stats_fake.GetXaxis().SetTitle('Unrolled 2D bins')
        ratio_stats_fake.GetXaxis().SetTitleOffset(0.5)
        ratio_stats_fake.GetYaxis().SetNdivisions(405)
        ratio_stats_fake.GetXaxis().SetLabelSize(3.* ratio_fake.GetXaxis().GetLabelSize())
        ratio_stats_fake.GetYaxis().SetLabelSize(3.* ratio_fake.GetYaxis().GetLabelSize())
        ratio_stats_fake.GetXaxis().SetTitleSize(3.* ratio_fake.GetXaxis().GetTitleSize())
        ratio_stats_fake.GetYaxis().SetTitleSize(3.* ratio_fake.GetYaxis().GetTitleSize())

        norm_stack_fake = ROOT.THStack('norm_stack', '')

        for key in hists_unrolled_fake.keys():
            if key == '%s_data'%k: continue
            vv = hists_unrolled_fake[key]
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
    
        c2.Modified()
        c2.Update()
        c2.SaveAs('multi_plots/%s/unrolled/fail_region/pdf/%s.pdf' %(label, k))
        c2.SaveAs('multi_plots/%s/unrolled/fail_region/png/%s.png' %(label, k))
    
    save_selection(label, preselection)
    save_weights(label, sample_names, weights)

        
        
            
