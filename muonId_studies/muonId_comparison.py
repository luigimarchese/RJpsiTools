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
n_files = 50


label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
if not os.path.isdir('plots'):
    os.mkdir('plots')
if not os.path.isdir('plots/%s' %label):
    os.mkdir('plots/%s' %label)

events_signal = ROOT.TChain("Events")
for f in files_signal[:n_files]:
    events_signal.AddFile(f)
events_bkg = ROOT.TChain("Events")
for f in files_bkg[:n_files]:
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

#variables = variables[:3]
c = ROOT.TCanvas("c","c",700, 700)
c.Draw()

#eff = array('d')
eff = ROOT.TH1F("eff","eff", len(variables),0,len(variables))
eff.SetTitle("; ; S/sqrt(B)")
for j,var in enumerate(variables):
    print("Processing variable "+var)
    h_sig = ROOT.TH1F("h_sig_"+var,"h_sig_"+var,n_bins,xmin,xmax)
    h_bkg = ROOT.TH1F("h_bkg_"+var,"h_bkg_"+var,n_bins,xmin,xmax)

    # selection for signal:
    # that the muon is a real Muon "abs(GenPart_pdgId[Muon_genPartIdx]) == 13"
    # that we have a candidate in the event "nBTo3Mu"
    # we choose the third muon "var +"[BTo3Mu_kIdx]""
    # the muon comes from a tau  "abs(GenPart_pdgId[GenPart_genPartIdxMother[Muon_genPartIdx]]) == 15  && abs(GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[Muon_genPartIdx]]]) == 541"

    events_signal.Draw(var + ">>h_sig_"+var,var +"[BTo3Mu_kIdx] && Muon_genPartIdx!=-1 && abs(GenPart_pdgId[Muon_genPartIdx]) == 13 && nBTo3Mu && abs(GenPart_pdgId[GenPart_genPartIdxMother[Muon_genPartIdx]]) == 15  && abs(GenPart_pdgId[GenPart_genPartIdxMother[GenPart_genPartIdxMother[Muon_genPartIdx]]]) == 541")

    # selection for fakes:
    # that the candidate exists "nBTo3Mu"
    # we choose the third muon  "var +"[BTo3Mu_kIdx]""
    # third muon is fake!" abs(GenPart_pdgId[Muon_genPartIdx]) != 15"
    events_bkg.Draw(var + ">>h_bkg_"+var,var +"[BTo3Mu_kIdx]  && nBTo3Mu && Muon_genPartIdx!=-1 &&  abs(GenPart_pdgId[Muon_genPartIdx]) != 15 ")

    leg=ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_sig,"signal","F")
    leg.AddEntry(h_bkg,"bkg","F")
    if h_sig.Integral() and h_bkg.Integral():
        h_sig.Scale(1./h_sig.Integral())
        h_bkg.Scale(1./h_bkg.Integral())
    elif h_bkg.Integral():
        h_bkg.Scale(1./h_bkg.Integral())
    elif h_sig.Integral():
        h_sig.Scale(1./h_sig.Integral())

    h_sig.SetMaximum(2. * max([h_sig.GetMaximum(), h_bkg.GetMaximum()]))
    h_sig.Draw("hist")
    h_bkg.SetLineColor(ROOT.kRed)
    h_bkg.Draw("sameHIST")
    leg.Draw("same");
    c.SaveAs("plots/"+label+"/histo_"+var+".png")
    print("signal integral",h_sig.Integral(),"bkg integral",h_bkg.Integral())
    print(h_sig.GetBinContent(2),h_bkg.GetBinContent(2))
    if(h_bkg.GetBinContent(2) == 0 and h_sig.GetBinContent(2) == 0):
        eff.SetBinContent(j+1,h_sig.GetBinContent(2))
    elif(h_bkg.GetBinContent(2) == 0):
        eff.SetBinContent(j+1,h_sig.GetBinContent(2)) #big value
    else:
        eff.SetBinContent(j+1,h_sig.GetBinContent(2)/math.sqrt(h_bkg.GetBinContent(2)))
    print(eff.GetBinContent(j+1))
c = ROOT.TCanvas("c","c",700, 700)
c.Draw()
leg=ROOT.TLegend(0.24,.67,.95,.90)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)

#x = array('d')
#for i in range(len(variables)):
#    x.append(i)
#graph = ROOT.TGraph(len(variables),x,eff)
#graph.Draw("ac*")
#xax = graph.GetXaxis()
#for i,name in enumerate(variables):
#    xax.SetBinLabel(i+1,name)

#c.Update()

for i in range(eff.GetNbinsX()):
    eff.GetXaxis().SetBinLabel(i+1,variables[i].split("Muon_")[1])

eff.GetXaxis().SetLabelSize(0.02)
eff.GetXaxis().LabelsOption("v")
eff.Draw("hist")
leg.Draw("same")
c.SaveAs("plots/"+label+"/final_plot.png")

output = ROOT.TFile.Open("plots/"+label+"/final_histo.root","recreate")
output.cd()
eff.Write()
output.Close()
