import ROOT
#from sklearn.metrics import roc_curve, roc_auc_score
from bokeh.palettes import all_palettes
from array import array
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

xmin = 0

f_signal = ROOT.TFile.Open("RJPsi_BcToJPsiMuMu_25Feb2021_500k_fastjet_mc_10614.root","r")
f_bkg = ROOT.TFile.Open("RJPsi_HbToJPsiMuMu_25Feb2021_500k_fastjet_mc_10614.root","r")

tree_signal = f_signal.Get("Events")
tree_bkg = f_bkg.Get("Events")

new_f_signal = ROOT.TFile("RJPsi_BcToJPsiMuMu_25Feb2021_500k_newbranches.root","RECREATE")

new_signal = tree_signal.CloneTree(0)

rho_corr_iso03_sig = np.array([0.])

rho_03_sig_branch = new_signal.Branch("Muon_rho_corr_iso03",rho_corr_iso03_sig,"rho_corr_iso03_sig/D")


def getAreaEff( eta, drcone ):
    aeff_dic = { '03' : 
              [ (1.000, 0.13),
              (1.479, 0.14),
              (2.000, 0.07),
              (2.200, 0.09),
              (2.300, 0.11),
              (2.400, 0.11),
              (2.500, 0.14) ],
           '04' : 
              [ (1.000, 0.208),
              (1.479, 0.209),
              (2.000, 0.115),
              (2.200, 0.143),
              (2.300, 0.183),
              (2.400, 0.194),
              (2.500, 0.261) ],
    }

    for i,eta_loop in enumerate(aeff_dic[drcone]):
        if(i == 0 and eta<=eta_loop[0]):
            return aeff_dic[drcone][0][1]
        if eta>eta_loop[0]:
            return aeff_dic[drcone][i-1][1]

#add rho correction branches
for i in range(tree_signal.GetEntries()):
    tree_signal.GetEntry(i)
    if(new_signal.nBTo3Mu):
        Aeff = getAreaEff(abs(new_signal.Muon_eta[new_signal.BTo3Mu_kIdx[0]]),'03')
        rho_corr_iso03_sig[0] = new_signal.Muon_raw_ch_pfiso03[0] + max(new_signal.Muon_raw_n_pfiso03[0] + new_signal.Muon_raw_pho_pfiso03[0] - new_signal.fixedGridRhoFastjetAll*Aeff,0)
    else:
        rho_corr_iso03_sig[0] = -99.
    new_signal.Fill()
print("Finito sign")


new_f_signal.Write()
new_f_signal.Close()
f_signal.Close()

new_f_bkg = ROOT.TFile("RJPsi_HbToJPsiMuMu_25Feb2021_500k_newbranches.root","RECREATE")
new_bkg = tree_bkg.CloneTree(0)
rho_corr_iso03_bkg = np.array([0.])
rho_03_bkg_branch = new_bkg.Branch("Muon_rho_corr_iso03",rho_corr_iso03_bkg,"rho_corr_iso03_bkg/D")

for i in range(tree_bkg.GetEntries()):
    tree_bkg.GetEntry(i)
    if(new_bkg.nBTo3Mu):
        Aeff = getAreaEff(abs(new_bkg.Muon_eta[new_bkg.BTo3Mu_kIdx[0]]),'03')
        rho_corr_iso03_bkg[0] = new_bkg.Muon_raw_ch_pfiso03[0] + max(new_bkg.Muon_raw_n_pfiso03[0] + new_bkg.Muon_raw_pho_pfiso03[0] - new_bkg.fixedGridRhoFastjetAll*Aeff,0)
    else:
        rho_corr_iso03_bkg[0] = -999.
       
    new_bkg.Fill()

new_f_bkg.Write()
new_f_bkg.Close()
f_bkg.Close()
