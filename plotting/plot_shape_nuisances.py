import ROOT
from samples_wf import sample_names
from officialStyle import officialStyle
from cmsstyle import CMS_lumi
import os
from histos import histos
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

def plot_shape_nuisances(histos_folder, variable, pf = 'pass', fakes = True, path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/plots_ul/', verbose = False):
    '''
    Function that plots the systematic shape uncertainties in a root file.
    It takes as input the 
    - histos_folder -> folder in which the root files is saved
    - variable -> which variable to plot
    - pf -> if the region is pass or fail (used to find the right datacard)
    - fakes -> if it comes from the analysis in which the fakes are from data, or if it's from an only pass region
    - path -> the path
    '''

    # output folder
    path_out_1 = path+histos_folder+'/shape_nuis'
    if not os.path.exists(path_out_1) : os.system('mkdir -p %s'%path_out_1)
    path_out_2 = path_out_1 + '/'+variable
    if not os.path.exists(path_out_2) : os.system('mkdir -p %s'%path_out_2)
    path_out = path_out_2 + '/'+pf
    if not os.path.exists(path_out) : os.system('mkdir -p %s'%path_out)


    # Input datacard path

    datacard_path = path+histos_folder+'/datacards/datacard_'+pf+'_'+variable+'.root'
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

    # Plot 
    c2 = ROOT.TCanvas('c2', '', 700, 700)
    c2.Draw()
    c2.cd()
    c2.SetTicks(True)
    c2.SetBottomMargin(0.15)
    #ROOT.gPad.SetLogx()    

    for sname in sample_names:
        # Only data and fakes don't have any shape nuisance
        if (sname != 'data' and sname != 'fakes'):
            if (fakes == True and sname == 'jpsi_x'):
                continue
            if verbose: print("Sample: ",sname)
            # all have ctau syst
            his_central = fin.Get(sname)
            xmin = his_central.GetBinLowEdge(1)
            xmax = his_central.GetBinLowEdge(his_central.GetNbinsX() + 1)
            nbins = his_central.GetNbinsX()
            
            histo_central = ROOT.TH1F(sname,sname, nbins, xmin, xmax)

            for i in range(1,his_central.GetNbinsX()+1):
                histo_central.SetBinContent(i,his_central.GetBinContent(i))
                histo_central.SetBinError(i,his_central.GetBinError(i))
        
            for syst in hammer_syst + ctau_syst + pu_syst:
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
                his_up = fin.Get(sname+'_'+syst+'Up')
                histo_up = ROOT.TH1F(sname+'_'+syst+'Up',sname+'_'+syst+'Up', nbins, xmin, xmax)
                for i in range(1,his_up.GetNbinsX()+1):
                    histo_up.SetBinContent(i,his_up.GetBinContent(i))
                    histo_up.SetBinError(i,his_up.GetBinError(i))

                maxx.append(histo_up.GetMaximum())
                his_down = fin.Get(sname+'_'+syst+'Down')
                histo_down = ROOT.TH1F(sname+'_'+syst+'Down',sname+'_'+syst+'Down', nbins, xmin, xmax)

                for i in range(1,his_down.GetNbinsX()+1):
                    histo_down.SetBinContent(i,his_down.GetBinContent(i))
                    histo_down.SetBinError(i,his_down.GetBinError(i))
                maxx.append(histo_down.GetMaximum())
                
                histo_central.SetTitle(sname+' '+syst+';'+histos[variable][1]+';events')
                histo_central.SetLineColor(ROOT.kBlack)
                histo_central.SetMarkerStyle(7)
                histo_central.SetMarkerColor(ROOT.kBlack)
                histo_up.SetLineColor(ROOT.kRed)
                histo_up.SetMarkerStyle(7)
                histo_up.SetMarkerColor(ROOT.kRed)
                histo_down.SetLineColor(ROOT.kGreen)
                histo_down.SetMarkerStyle(7)
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
                
                #CMS_lumi(c2, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
                
                c2.Modified()
                c2.Update()
                
                c2.SaveAs(path_out+"/"+sname+'_'+syst+'.png')

if __name__ == "__main__":

    histos_folder = '03May2021_15h16m29s'
    variable = 'decay_time_ps'
    plot_shape_nuisances(histos_folder, variable, verbose = True)
    
