'''
Difference with v1:
- fixed bugs to be compatible with showplots_v6 and showplots_v7
  - new channels definition (ch# instead of pass and fail); addition of 2 new categories
'''

import ROOT
from officialStyle import officialStyle
from samples import sample_names as complete_sample_names
from cmsstyle import CMS_lumi
import os
from histos import histos
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

sfrange = 16

def plot_shape_nuisances(histos_folder, variable, channel, sample_names=complete_sample_names, plot3d = False, fakes = True, path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/plots_ul/', compute_sf = False, verbose = False, compute_sf_onlynorm = False):
    '''
    Function that plots the systematic shape uncertainties in a root file.
    It takes as input the 
    - histos_folder -> folder in which the root files is saved
    - variable -> which variable to plot
    - channel -> if the region is ch1,ch2,ch3,ch4
    - fakes -> if it comes from the analysis in which the fakes are from data, or if it's from an only pass region
    - path -> the path
    '''
    
    print("#######################  "+channel+ "  #####################")
    # output folder
    path_out_1 = path+histos_folder+'/shape_nuis'
    if not os.path.exists(path_out_1) : os.system('mkdir -p %s'%path_out_1)
    path_out_2 = path_out_1 + '/'+variable
    if not os.path.exists(path_out_2) : os.system('mkdir -p %s'%path_out_2)
    path_out = path_out_2 + '/'+channel
    if not os.path.exists(path_out) : os.system('mkdir -p %s'%path_out)


    datacard_path = path+histos_folder+'/datacards/datacard_'+channel+'_'+variable+'.root'
    if verbose : print("Opening datacard: "+datacard_path+"")
    fin = ROOT.TFile.Open(datacard_path,'r')
    hammer_syst = ['bglvar_e0',
                   'bglvar_e1',
                   'bglvar_e2',
                   'bglvar_e3',
                   'bglvar_e4',
                   'bglvar_e5',
                   'bglvar_e6',
                   'bglvar_e7',
                   'bglvar_e8',
                   'bglvar_e9',
                   'bglvar_e10',
               ]

    ctau_syst = ['ctau',]
    pu_syst = ['puWeight']

    his_tmp = fin.Get('jpsi_x_mu_'+channel)
    nbins = his_tmp.GetNbinsX()

    bbb_syst = ['bbb'+str(i)+channel for i in range(1,nbins+1)]

    total_syst = hammer_syst + ctau_syst + pu_syst + bbb_syst 

    # Don't compute these because they are approximable to a normalisation nuisance
    # Keep the code in case we need to compute average and max again
    if compute_sf:
        sf_reco_syst = ['sfReco_'+str(i) for i in range(sfrange)]
        sf_id_syst = ['sfId_'+str(i) for i in range(sfrange)]
        total_syst = total_syst + sf_reco_syst + sf_id_syst
    
    if compute_sf_onlynorm:
        print("==============================================================")
        print("==== Computing Reco and Id SF uncertainties for "+variable+" ========")
        print("==============================================================")
        sf_reco_syst = ['sfReco']
        sf_id_syst = ['sfId']
        total_syst = total_syst + sf_reco_syst + sf_id_syst

    # Plot 
    c3 = ROOT.TCanvas('c3', '', 700, 700)
    c3.Draw()
    c3.cd()
    c3.SetTicks(True)
    c3.SetBottomMargin(0.15)
    #ROOT.gPad.SetLogx()    

    for sname in sample_names:
        # Only data and fakes don't have any shape nuisance
        if (sname != 'data' and sname != 'fakes'):

            # For the computation of scale factors, I need to reset the value for each sample!
            if compute_sf:
                reco = []
                id = []
            if (fakes == True and sname == 'jpsi_x'):
                continue
            if verbose: print("Sample: ",sname)
            # all have ctau syst
            his_central = fin.Get(sname+'_'+channel)
            xmin = his_central.GetBinLowEdge(1)
            xmax = his_central.GetBinLowEdge(his_central.GetNbinsX() + 1)
            nbins = his_central.GetNbinsX()
            
            histo_central = ROOT.TH1F(sname,sname, nbins, xmin, xmax)

            for i in range(1,his_central.GetNbinsX()+1):
                histo_central.SetBinContent(i,his_central.GetBinContent(i))
                histo_central.SetBinError(i,his_central.GetBinError(i))
        
            for syst in total_syst:
                if verbose: print("Plotting variable "+syst+"for dataset "+sname)
                maxx = []

                # Only jpsi tau and jpsi mu have the form factor systematics
                if syst in hammer_syst:
                    if sname != 'jpsi_tau' and sname != 'jpsi_mu':
                        continue

                # Only Bc datasets have the ctau systematics
                if syst in ctau_syst:
                    if sname == 'jpsi_x' or sname == 'jpsi_x_mu':
                        continue
                if syst in bbb_syst:
                    if sname != 'jpsi_x_mu':
                        continue
                his_up = fin.Get(sname+'_'+syst+'Up_'+channel)
                histo_up = ROOT.TH1F(sname+'_'+syst+'Up',sname+'_'+syst+'Up', nbins, xmin, xmax)
                for i in range(1,his_up.GetNbinsX()+1):
                    histo_up.SetBinContent(i,his_up.GetBinContent(i))
                    histo_up.SetBinError(i,his_up.GetBinError(i))

                maxx.append(histo_up.GetMaximum())
                his_down = fin.Get(sname+'_'+syst+'Down_'+channel)
                histo_down = ROOT.TH1F(sname+'_'+syst+'Down',sname+'_'+syst+'Down', nbins, xmin, xmax)

                for i in range(1,his_down.GetNbinsX()+1):
                    histo_down.SetBinContent(i,his_down.GetBinContent(i))
                    histo_down.SetBinError(i,his_down.GetBinError(i))
                maxx.append(histo_down.GetMaximum())
                
                if path == '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/multi_plots/':
                    histo_central.SetTitle(';Unrolled 2D bins;events')
                else:
                    histo_central.SetTitle(';'+histos[variable][1]+';events')
                    
                histo_central.SetLineColor(ROOT.kBlack)
                histo_central.SetMarkerStyle(8)
                histo_central.SetMarkerColor(ROOT.kBlack)
                histo_up.SetLineColor(ROOT.kRed)
                histo_up.SetMarkerStyle(8)
                histo_up.SetMarkerColor(ROOT.kRed)
                histo_down.SetLineColor(ROOT.kGreen)
                histo_down.SetMarkerStyle(8)
                histo_down.SetMarkerColor(ROOT.kGreen)
                histo_central.SetMaximum(1.5*max(maxx))
                histo_central.Draw("ep")
                histo_up.Draw("ep same")
                histo_down.Draw("ep same")
                
                leg = ROOT.TLegend(0.5,.75,.95,.90)
                leg.SetBorderSize(0)
                leg.SetFillColor(0)
                leg.SetFillStyle(0)
                leg.SetTextFont(42)
                leg.SetTextSize(0.035)
                leg.AddEntry(histo_central, 'central value',  'EP')
                leg.AddEntry(histo_up, 'up value',  'EP')
                leg.AddEntry(histo_down, 'down value',  'EP')
                leg.Draw('same')
                
                CMS_lumi(c3, 4, 0, cmsText = 'CMS', extraText = ' Work in Progress', lumi_13TeV = 'L = 59.7 fb^{-1}', verbose = False)
                
                c3.Modified()
                c3.Update()
                
                c3.SaveAs(path_out+"/"+sname+'_'+syst+'.png')
                c3.SaveAs(path_out+"/"+sname+'_'+syst+'.pdf')
                
                # COmpute the scale factors for muons
                if compute_sf:
                    if syst in sf_reco_syst or syst in sf_id_syst:
                        histo_up.Divide(histo_central)
                        histo_up.Draw("ep")
                        # average normalisation nuisance
                        avg = 0.
                        for b in range(1,histo_up.GetNbinsX()+1):
                            avg += histo_up.GetBinContent(b)
                        avg = avg/histo_up.GetNbinsX()
                        if syst in sf_reco_syst:
                            reco.append(avg)
                        elif syst in sf_id_syst:
                            id.append(avg)
                        avg_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
                        avg_value.AddText('avg = %.10f' %avg)
                        avg_value.SetFillColor(0)
                        avg_value.Draw('EP same')

                        c3.Modified()
                        c3.Update()
                    
                        c3.SaveAs(path_out+"/"+sname+'_'+syst+'_RATIO.png')
    
                        # for each sample I have a different value of scale factor
                        #if compute_sf:    
                        #    if syst in sf_reco_syst or syst in sf_id_syst:
                        #        print(sname," reco:",max(reco)," id:", max(id))
            
                # COmpute the scale factors for muons as normalisations
                if compute_sf_onlynorm:
                    if syst in sf_reco_syst or syst in sf_id_syst:
                        nuisance = histo_up.Integral()/histo_central.Integral()
                        if syst in sf_reco_syst:
                            print(sname," reco:",nuisance)
                        elif syst in sf_id_syst:
                            print(sname," id:",nuisance)



if __name__ == "__main__":

    histos_folder = '03Dec2021_13h45m02s'
    variable = 'Bmass'
    plot_shape_nuisances(histos_folder, variable,'ch3', [],verbose = False, compute_sf_onlynorm = True)
    
