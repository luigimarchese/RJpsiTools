# Compare different MC shapes
import numpy as np
import ROOT
from array import array
from root_pandas import read_root, to_root
from glob import glob

# cms libs
from samples import sample_names_explicit_jpsimother_compressed as sample_names
from cmsstyle import CMS_lumi
from officialStyle import officialStyle

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
officialStyle(ROOT.gStyle, ROOT.TGaxis)

tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/Rjpsi_inspector_bc_mu_01Dec21_v1/'

c1 = ROOT.TCanvas()

slope = 0.008

df =read_root('%s/inspector_bc_mu_merged.root'%(tree_dir), "tree")#, where = preselection) 
df.index= [i for i in range(len(df))]

nbins = 40
bins_array_left = [0.,6.]
bins_array_central   = [i*((25-6)/nbins)+6 for i in range(1,nbins+1)]
#bins_array_central_3   = [i*((35-20)/20)+6 for i in range(1,20+1)]
bins_array_central_2 = [i*((50-25)/20)+25 for i in range(1,21)]

bins_array_right = [150.]

bins_array = bins_array_left + bins_array_central + bins_array_central_2 + bins_array_right
#bins_array = bins_array_left + bins_array_central_2  
print(bins_array)

nbins_final = len(bins_array)
mc_histo = ROOT.TH1F("mc","mc",nbins_final-1,array('d',bins_array))
mc_histo_up = ROOT.TH1F("mc_up","mc_up",nbins_final-1,array('d',bins_array))
mc_histo_down = ROOT.TH1F("mc_up","mc_up",nbins_final-1,array('d',bins_array))
c1.Draw()
c1.cd()

# MC histo

up_branch = (1+slope) * df['bhad_pt']
down_branch = (1-slope) * df['bhad_pt']
for event in df['bhad_pt']:
    mc_histo.Fill(event)
for event in up_branch:
    mc_histo_up.Fill(event)
for event in down_branch:
    mc_histo_down.Fill(event)

mc_histo.SetTitle(";B_{c} gen p_{T} GeV;events")
mc_histo.SetLineColor(ROOT.kRed)
mc_histo.SetFillColor(ROOT.kWhite)
mc_histo_up.SetLineColor(ROOT.kBlue)
mc_histo_up.SetFillColor(ROOT.kWhite)
mc_histo_down.SetLineColor(ROOT.kGreen)
mc_histo_down.SetFillColor(ROOT.kWhite)
#mc_histo.SetMaximum(1.2* max(mc_histo.GetMaximum(),mc_histo_corr.GetMaximum()))
mc_histo.Draw("hist")
mc_histo_up.Draw("hist same")
mc_histo_down.Draw("hist same")

#    mc_histo_corr.Draw("hist same")
CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = 'L = 59.7 fb^{-1}')
#c1.SetLogy()
c1.Update()
c1.SaveAs("plots/correction.pdf")

mc_histo_up.Divide(mc_histo)
mc_histo_down.Divide(mc_histo)
mc_histo.Divide(mc_histo)
mc_histo_up.Draw("hist ")
mc_histo.Draw("hist same")
mc_histo_down.Draw("hist same")

c1.Update()
c1.SaveAs("plots/correction_ratio.pdf")

# save root file
fout = ROOT.TFile.Open("correction.root","RECREATE")
fout.cd()

mc_histo_up.Write()
mc_histo_down.Write()
mc_histo.Write()
fout.Close()

# Compute the weights
for k in sample_names:
    print(k)
    sample_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/'
    df =read_root('%s/%s_with_mc_corrections.root'%(sample_dir,k), "BTo3Mu")#, where = preselection) 
    df_final = df.copy()
    if k == 'data' or 'jpsi_x_mu' in k:
        df_final.to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/%s_with_mc_corrections.root'%(k),key='BTo3Mu', store_index=False)
        continue
    df_final['bc_mc_correction_weight_central'] = [1 for i in range(len(df))]
    weights_up = []
    weights_down = []
    for i in range(1,mc_histo.GetNbinsX()+1):
        if mc_histo.GetBinContent(i)==0:
            weights_up.append(1.)
            weights_down.append(1.)
        else:
            weights_up.append(mc_histo_up.GetBinContent(i)/mc_histo.GetBinContent(i))
            weights_down.append(mc_histo_down.GetBinContent(i)/mc_histo.GetBinContent(i))
        
    #Save the weights
    df_final_up = []
    df_final_down = []

    for i in range(len(df_final)):
        # for each event find which bin it belongs to
        binx = max(1, min(mc_histo.GetNbinsX(), mc_histo.GetXaxis().FindBin(df_final['bc_gen_pt'][i])));
        # find the associated weight
        df_final_up.append(weights_up[binx-1])
        df_final_down.append(weights_down[binx-1])
    
    df_final['bc_mc_correction_weight_up_0p8'] = df_final_up
    df_final['bc_mc_correction_weight_down_0p8'] = df_final_down
    #df_final['bc_mc_correction_weight_up_norm_v2'] = df_final_up/df_final['bc_mc_correction_weight_up_v2'].mean()
    #df_final['bc_mc_correction_weight_down_norm_v2'] = df_final_down/df_final['bc_mc_correction_weight_down_v2'].mean()    
    df_final.to_root('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_Dec2021/%s_with_mc_corrections.root'%(k),key='BTo3Mu', store_index=False)

