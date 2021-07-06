import ROOT
#from sklearn.metrics import roc_curve, roc_auc_score
from bokeh.palettes import all_palettes
from array import array
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

xmin = 0
n_cuts = 20

#f_signal = ROOT.TFile.Open("RJPsi_BcToJPsiMuMu_25Feb2021_500k_fastjet_mc_10614.root","r")
#f_bkg = ROOT.TFile.Open("RJPsi_HbToJPsiMuMu_25Feb2021_500k_fastjet_mc_10614.root","r")
f_signal = ROOT.TFile.Open("RJPsi_BcToJPsiMuMu_25Feb2021_500k_newbranches.root","r")
f_bkg = ROOT.TFile.Open("RJPsi_HbToJPsiMuMu_25Feb2021_500k_newbranches.root","r")

tree_signal = f_signal.Get("Events")
tree_bkg = f_bkg.Get("Events")

#loop sulle diverse variabili
#usa dataframes (per ora loop sugli eventi perche non li ho fatto datamframes)
variables = {
    'Muon_rho_corr_iso03':{'color':ROOT.kViolet+7,'xmax':1.5},
    'BTo3Mu_k_iso03':{'color':ROOT.kBlue,'xmax':1},
    'BTo3Mu_k_iso04':{'color':ROOT.kPink+1,'xmax':1},
    'Muon_db_corr_iso03':{'color':ROOT.kBlack,'xmax':3},
    'Muon_db_corr_iso04':{'color':ROOT.kYellow,'xmax':3},
    'Muon_raw_ch_pfiso03':{'color':ROOT.kTeal,'xmax':3},
    'Muon_raw_ch_pfiso04':{'color':ROOT.kGray,'xmax':3},
    'Muon_raw_n_pfiso03':{'color':ROOT.kMagenta,'xmax':1.2},
    'Muon_raw_n_pfiso04':{'color':ROOT.kRed-2,'xmax':2},
    'Muon_raw_pho_pfiso03':{'color':ROOT.kRed,'xmax':1},
    'Muon_raw_pho_pfiso04':{'color':ROOT.kCyan+2,'xmax':1.2},
    'Muon_raw_pu_pfiso03':{'color':ROOT.kViolet,'xmax':3},
    'Muon_raw_pu_pfiso04':{'color':ROOT.kGreen,'xmax':5},
    'Muon_raw_trk_iso03':{'color':ROOT.kOrange+7,'xmax':3.5},
    'Muon_raw_trk_iso05':{'color':ROOT.kOrange-3,'xmax':3.5},

}

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
'''
#add rho correction branches
new_signal = tree_signal.CloneTree(0)
rho_corr_iso03 = np.array([0.])
sign_newbranch = new_signal.Branch("Muon_rho_corr_iso03",rho_corr_iso03,'rho_corr/D')
for i in range(tree_signal.GetEntries()):
    tree_signal.GetEntry(i)
    if(new_signal.nBTo3Mu):
        Aeff = getAreaEff(abs(new_signal.Muon_eta[new_signal.BTo3Mu_kIdx[0]]),'03')
        print(Aeff)
        #print(new_signal.Muon_eta[new_signal.BTo3Mu_kIdx[0]])
        rho_corr_iso03[0] = new_signal.Muon_raw_ch_pfiso03[0] + max(new_signal.Muon_raw_n_pfiso03[0] + new_signal.Muon_raw_pho_pfiso03[0] - new_signal.fixedGridRhoFastjetAll*Aeff,0)
    else:
        print("no")
        rho_corr_iso03[0] = -99.
    new_signal.Fill()
print("Finito")
new_bkg = tree_bkg.CloneTree(0)
rho_corr_iso03_bkg = np.array([0.])
new_bkg.Branch("Muon_rho_corr_iso03",rho_corr_iso03_bkg,'rho_corr/D')
for i in range(tree_bkg.GetEntries()):
    tree_bkg.GetEntry(i)
    if(new_bkg.nBTo3Mu):
        #print(new_bkg.Muon_eta[tree_signal.BTo3Mu_kIdx])
        Aeff = getAreaEff(abs(new_bkg.Muon_eta[new_bkg.BTo3Mu_kIdx[0]]),'03')
        rho_corr_iso03_bkg[0] = new_bkg.Muon_raw_ch_pfiso03[0] + max(new_bkg.Muon_raw_n_pfiso03[0] + new_bkg.Muon_raw_pho_pfiso03[0] - new_bkg.fixedGridRhoFastjetAll*Aeff,0)
    else:
        rho_corr_iso03_bkg[0] = -999.
       
    new_bkg.Fill()
'''
c = ROOT.TCanvas("c","c",700, 700)
c.Draw()
leg = ROOT.TLegend(0.10,.1,.75,.45)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.028)
mg = ROOT.TMultiGraph()
xy=array('d')
for j,var in enumerate(variables):
    print("Processing variable ",var)
    h_sig = ROOT.TH1F("h_sig_"+var,"h_sig_"+var,n_cuts,xmin,variables[var]['xmax'])
    h_bkg = ROOT.TH1F("h_bkg_"+var,"h_bkg_"+var,n_cuts,xmin,variables[var]['xmax'])

    #compute variable
    '''    if(var == 'Muon_rho_corr_iso03'):
        for i in range(tree_signal.GetEntries()):
            tree_signal.GetEntry(i)
            if(tree_signal.nBTo3Mu and tree_signal.Muon_genPartIdx[0]!=-1 and abs(tree_signal.GenPart_pdgId[tree_signal.Muon_genPartIdx[0]]) == 13):
                Aeff = getAreaEff(abs(tree_signal.Muon_eta[tree_signal.BTo3Mu_kIdx[0]]),'03')
                #print("start",Aeff,abs(tree_signal.Muon_eta[tree_signal.BTo3Mu_kIdx[0]]))
                #print(tree_signal.Muon_raw_ch_pfiso03[0])
                #print(tree_signal.Muon_raw_n_pfiso03[0])
                #print(tree_signal.Muon_raw_pho_pfiso03[0])
                #print(tree_signal.fixedGridRhoFastjetAll)
                rho_corr_iso03 = tree_signal.Muon_raw_ch_pfiso03[0] + max(tree_signal.Muon_raw_n_pfiso03[0] + tree_signal.Muon_raw_pho_pfiso03[0] - tree_signal.fixedGridRhoFastjetAll*Aeff,0.)
                h_sig.Fill(rho_corr_iso03/tree_signal.Muon_pt[tree_signal.BTo3Mu_kIdx[0]])
        print("Signal finished...")
        for i in range(tree_bkg.GetEntries()):
            tree_bkg.GetEntry(i)
            if(tree_bkg.nBTo3Mu):
                Aeff = getAreaEff(abs(tree_bkg.Muon_eta[tree_bkg.BTo3Mu_kIdx[0]]),'03')
                rho_corr_iso03 = tree_bkg.Muon_raw_ch_pfiso03[0] + max(tree_bkg.Muon_raw_n_pfiso03[0] + tree_bkg.Muon_raw_pho_pfiso03[0] - tree_bkg.fixedGridRhoFastjetAll*Aeff,0)
                h_bkg.Fill(rho_corr_iso03/tree_bkg.Muon_pt[tree_bkg.BTo3Mu_kIdx[0]])
        print("Bkg finished...")
    else:'''
    if('BTo3Mu' in var):
        tree_signal.Draw(var + "/Muon_pt[BTo3Mu_kIdx]>>h_sig_"+var,"Muon_genPartIdx!=-1 && abs(GenPart_pdgId[Muon_genPartIdx]) == 13 && nBTo3Mu ")
        #tree_bkg.Draw(var + ">>h_bkg_"+var,"Muon_genPartIdx!=-1 && abs(GenPart_pdgId[Muon_genPartIdx]) != 13 && nBTo3Mu")
        tree_bkg.Draw(var + "/Muon_pt[BTo3Mu_kIdx]>>h_bkg_"+var," nBTo3Mu")
    if('Muon' in var): # In Muon collection we need to choose the third muon
        tree_signal.Draw(var + "/Muon_pt[BTo3Mu_kIdx]>>h_sig_"+var, var +"[BTo3Mu_kIdx] && Muon_genPartIdx!=-1 && abs(GenPart_pdgId[Muon_genPartIdx]) == 13 && nBTo3Mu ")
        #        tree_bkg.Draw(var + ">>h_bkg_"+var,var +"[BTo3Mu_kIdx] && Muon_genPartIdx!=-1 && nBTo3Mu && abs(GenPart_pdgId[Muon_genPartIdx]) != 13 ")
        tree_bkg.Draw(var + "/Muon_pt[BTo3Mu_kIdx]>>h_bkg_"+var,var +"[BTo3Mu_kIdx]  && nBTo3Mu  ")
    h_sig.Draw("hist")
    h_bkg.SetLineColor(ROOT.kRed)
    h_bkg.Draw("sameHIST")
    c.SaveAs("plot_"+var+".png")
    print("signal integral",h_sig.Integral(),"bkg integral",h_bkg.Integral())
    eff_sig = array('d')
    rej_bkg = array('d')
    for i in range(n_cuts):
        point = i * (variables[var]['xmax']-xmin)/n_cuts + xmin 
        if(point == xmin): 
            eff_sig.append(0)
            rej_bkg.append(1.)
        else:
            eff_sig.append(h_sig.Integral(1,h_sig.GetXaxis().FindBin(point))/h_sig.Integral())
            eff_bkg = h_bkg.Integral(1,h_bkg.GetXaxis().FindBin(point))/h_bkg.Integral()
            rej_bkg.append(1 - eff_bkg)
    graph = ROOT.TGraph(n_cuts,rej_bkg,eff_sig)
    graph.SetTitle(var)
    graph.SetMarkerColor(variables[var]['color'])
    graph.SetLineColor(variables[var]['color'])
    mg.Add(graph)
    leg.AddEntry(graph,var,'L')
    xy.append(point)
graph = ROOT.TGraph(n_cuts,xy, xy)
#mg.Add(graph)
mg.SetTitle("; bkg rejection;signal efficiency")
mg.Draw("ac*")
leg.Draw()

#c.BuildLegend()
c.SaveAs("plot_allroc.png")
