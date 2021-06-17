'''
Script to study the effect of diffent IDs of muons
in getting rid of fakes in the pass and fail definition
'''
from datetime import datetime
import ROOT
from bokeh.palettes import all_palettes
from array import array
import numpy as np
import math
from BcToJPsiMuMu_files_path_2021Apr20 import files as files_signal
from Hb_files import files as files_bkg
import os

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)
#ROOT.EnableImplicitMT()

xmin = 0.
xmax = 2.
n_bins = 2
n_files_signal = 2
n_files_bkg = 10


label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
if not os.path.isdir('plots'):
    os.mkdir('plots')
if not os.path.isdir('plots/%s' %label):
    os.mkdir('plots/%s' %label)

events_signal = ROOT.TChain("Events")
for f in files_signal[:n_files_signal]:
    events_signal.AddFile(f)
events_bkg = ROOT.TChain("Events")
for f in files_bkg[:n_files_bkg]:
    events_bkg.AddFile(f)

variables = [
    'Muon_looseId',
    'Muon_mediumId',
    'Muon_mediumpromptId',
    'Muon_tightId',
    'Muon_globalHighPtId',
    'Muon_trkHighPtId',
    'Muon_pfIsoVeryLooseId',
    'Muon_pfIsoLooseId',
    'Muon_pfIsoMediumId',
    'Muon_pfIsoTightId',
    'Muon_pfIsoVeryTightId',
    'Muon_pfIsoVeryVeryTightId',
    'Muon_tkIsoLooseId',
    'Muon_tkIsoTightId',
    'Muon_softId',
    'Muon_softMvaId',
    'Muon_mvaLooseId',
    'Muon_mvaTightId',
    'Muon_mvaMediumId',
    'Muon_miniIsoLooseId',
    'Muon_miniIsoMediumId',
    'Muon_miniIsoTightId',
    'Muon_miniIsoVeryTightId',
    'Muon_triggerLooseId',
    'Muon_inTimeMuonId',
    'Muon_multiIsoLooseId',
    'Muon_multiIsoMediumId',
    'Muon_puppiIsoLooseId',
    'Muon_puppiIsoMediumId',
    'Muon_puppiIsoTightId',
    'Muon_mvaVTightId',
    'Muon_mvaVVTightId',
    'Muon_lowPtMvaLooseId',
    'Muon_lowPtMvaMediumId',
]

preselection = ' && '.join([
    'Muon_pt[BTo3Mu_mu1Idx]>4',
    'Muon_pt[BTo3Mu_mu1Idx]>4',
    'Muon_pt[BTo3Mu_kIdx]>2.5',
    'abs(Muon_eta[BTo3Mu_mu1Idx])<2.5',
    'abs(Muon_eta[BTo3Mu_mu2Idx])<2.5',
    'abs(Muon_eta[BTo3Mu_kIdx])<2.5',
    'BTo3Mu_bodies3_svprob>1e-4',
    'BTo3Mu_jpsivtx_svprob>1e-2',
    'BTo3Mu_mass<6.3',
    'Muon_mediumId[BTo3Mu_mu1Idx]>0.5',
    'Muon_mediumId[BTo3Mu_mu2Idx]>0.5',
    'abs(BTo3Mu_mu1_dz - BTo3Mu_mu2_dz)<0.2',
    'abs(BTo3Mu_mu1_dz - BTo3Mu_k_dz)<0.2',
    'abs(BTo3Mu_mu2_dz - BTo3Mu_k_dz)<0.2',
    'abs(BTo3Mu_mu1_dxy)<0.05',
    'abs(BTo3Mu_mu2_dxy)<0.05',
    'abs(BTo3Mu_k_dxy)<0.05',
    #jpsi mass
    # dr12;dr13;dr23
    'Muon_isMuonFromJpsi_dimuon0Trg[BTo3Mu_mu1_dxy] >0.5',
    'Muon_isMuonFromJpsi_dimuon0Trg[BTo3Mu_mu2_dxy] >0.5',
    'Muon_isDimuon0Trg[BTo3Mu_k_dxy]>0.5',
])

#variables = variables[:3]
c = ROOT.TCanvas("c","c",700, 700)
c.Draw()

#eff = array('d')
eff = ROOT.TH1F("eff","eff", len(variables),0,len(variables))
eff.SetTitle("; ; S/sqrt(B)")
eff2 = ROOT.TH1F("eff2","eff2", len(variables),0,len(variables))
eff2.SetTitle("; ; S/B")
eff3 = ROOT.TH1F("eff3","eff3", len(variables),0,len(variables))
eff3.SetTitle("; ; S/sqrt(S+B)")
for j,var in enumerate(variables):
    pass_selection = ' && '.join([
        var +"[BTo3Mu_kIdx] > 0.5 ",
        "Muon_db_corr_iso03_rel[BTo3Mu_kIdx]<0.2"
    ])

    fail_selection = "(!("+pass_selection+"))"

    print("Processing variable "+var)
    h_sig = ROOT.TH1F("h_sig_"+var,"h_sig_"+var,n_bins,xmin,xmax)
    h_bkg = ROOT.TH1F("h_bkg_"+var,"h_bkg_"+var,n_bins,xmin,xmax)

    # selection for signal:
    # that the muon is a real Muon "abs(GenPart_pdgId[Muon_genPartIdx]) == 13"
    # that we have a candidate in the event "nBTo3Mu"
    # we choose the third muon "var +"[BTo3Mu_kIdx]""
    # the muon comes from a tau  "abs(GenPart_pdgId[GenPart_genPartIdxMother[Muon_genPartIdx]]) == 15  && abs(GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[Muon_genPartIdx]]]) == 541"
    signal_pass = events_signal.GetEntries("Muon_genPartIdx[BTo3Mu_kIdx]!=-1 && abs(GenPart_pdgId[Muon_genPartIdx[BTo3Mu_kIdx]]) == 13 && nBTo3Mu && abs(GenPart_pdgId[GenPart_genPartIdxMother[Muon_genPartIdx[BTo3Mu_kIdx]]]) == 15  && abs(GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[Muon_genPartIdx[BTo3Mu_kIdx]]]]) == 541 && " + preselection + ' && ' + pass_selection)
    signal_fail = events_signal.GetEntries(" Muon_genPartIdx[BTo3Mu_kIdx]!=-1 && abs(GenPart_pdgId[Muon_genPartIdx[BTo3Mu_kIdx]]) == 13 && nBTo3Mu && abs(GenPart_pdgId[GenPart_genPartIdxMother[Muon_genPartIdx[BTo3Mu_kIdx]]]) == 15  && abs(GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[Muon_genPartIdx[BTo3Mu_kIdx]]]]) == 541 && " + preselection + ' && ' + fail_selection)
    
    # selection for fakes:
    # that the candidate exists "nBTo3Mu"
    # we choose the third muon  "var +"[BTo3Mu_kIdx]""
    # third muon is fake!" abs(GenPart_pdgId[Muon_genPartIdx]) != 13" --> but we don't write this because it's enough to write Muon_genPartIdx[BTo3Mu_kIdx] == -1
    
    bkg_pass = events_bkg.GetEntries("nBTo3Mu && Muon_genPartIdx[BTo3Mu_kIdx]==-1 && " + preselection + ' && ' +pass_selection)
    bkg_fail = events_bkg.GetEntries("nBTo3Mu && Muon_genPartIdx[BTo3Mu_kIdx]==-1  && " + preselection+ ' && ' +fail_selection)
    
    print(signal_pass,signal_fail)
    print(bkg_pass,bkg_fail)
    
    signal_tot = signal_pass + signal_fail
    bkg_tot = bkg_pass+bkg_fail

    signal_pass = signal_pass/signal_tot
    signal_fail = signal_fail/signal_tot
    bkg_pass = bkg_pass/bkg_tot
    bkg_fail = bkg_fail/bkg_tot

    if bkg_pass == 0:
        eff.SetBinContent(j+1,0)
        eff2.SetBinContent(j+1,0)
        eff3.SetBinContent(j+1,0)
    else:
        eff.SetBinContent(j+1,signal_pass/math.sqrt(bkg_pass))
        eff2.SetBinContent(j+1,signal_pass/bkg_pass)
        eff3.SetBinContent(j+1,signal_pass/math.sqrt(signal_pass + bkg_pass))
    

c = ROOT.TCanvas("c","c",700, 700)
c.Draw()
leg=ROOT.TLegend(0.24,.67,.95,.90)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)


for i in range(eff.GetNbinsX()):
    eff.GetXaxis().SetBinLabel(i+1,variables[i].split("Muon_")[1])
    eff2.GetXaxis().SetBinLabel(i+1,variables[i].split("Muon_")[1])
    eff3.GetXaxis().SetBinLabel(i+1,variables[i].split("Muon_")[1])

eff.GetXaxis().SetLabelSize(0.02)
eff.GetXaxis().LabelsOption("v")
eff.Draw("hist")
leg.Draw("same")
c.SaveAs("plots/"+label+"/final_plot.png")

eff2.GetXaxis().SetLabelSize(0.02)
eff2.GetXaxis().LabelsOption("v")
eff2.Draw("hist")
leg.Draw("same")
c.SaveAs("plots/"+label+"/final_plot2.png")

eff3.GetXaxis().SetLabelSize(0.02)
eff3.GetXaxis().LabelsOption("v")
eff3.Draw("hist")
leg.Draw("same")
c.SaveAs("plots/"+label+"/final_plot3.png")

output = ROOT.TFile.Open("plots/"+label+"/final_histo.root","recreate")
output.cd()
eff.Write()
eff2.Write()
eff3.Write()
output.Close()
