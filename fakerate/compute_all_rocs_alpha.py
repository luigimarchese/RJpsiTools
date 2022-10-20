import ROOT
from root_pandas import read_root, to_root
from selections_for_fakerate import preprepreselection,triggerselection, etaselection
from new_branches_pandas import to_define
from array import array
from histos_nordf import histos
from bokeh.palettes import viridis, all_palettes
import pandas as pd
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

# mc
data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_nopresel_withpresel_v2_withnn_withidiso.root'
mc_path = []
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_mu_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_tau_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic0_mu_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic1_mu_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/chic2_mu_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_hc_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/hc_mu_nopresel_withpresel_v2_withnn_withidiso.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/psi2s_mu_nopresel_withpresel_v2.root')
mc_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/psi2s_tau_nopresel_withpresel_v2_withnn_withidiso.root')
jpsix_path = []
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bzero_nopresel_withpresel_v2_withnn_withidiso.root')
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bplus_nopresel_withpresel_v2_withnn_withidiso.root')
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_bzero_s_nopresel_withpresel_v2_withnn_withidiso.root')
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_sigma_nopresel_withpresel_v2_withnn.root')
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_lambdazero_b_nopresel_withpresel_v2_withnn_withidiso.root')
jpsix_path.append('/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/jpsi_x_mu_from_xi_nopresel_withpresel_v2_withnn_withidiso.root')

# mediumI & fail iso region for training
prepreselection = preprepreselection + "&" +triggerselection +"&"+etaselection +" & k_mediumID<0.5 & k_raw_db_corr_iso03_rel>0.2 & Bpt_reco<80"
data = read_root(data_path, 'BTo3Mu', where=prepreselection )
data = to_define(data)
data['target']= [0 for i in range(len(data.Bmass))]
mc = []
for mc_p in mc_path:
    print(mc_p)
    mcc=read_root(mc_p, 'BTo3Mu', where=prepreselection + " & (abs(k_genpdgId)==13)")
    mcc = to_define(mcc)
    mcc['target']= [1 for i in range(len(mcc.Bmass))]
    mc.append(mcc)

mcjpsix = []
for mc_p in jpsix_path:
    print(mc_p)
    mcc=read_root(mc_p, 'BTo3Mu', where=prepreselection + " & (abs(k_genpdgId)==13)")
    mcc = to_define(mcc)
    mcc['target']= [1 for i in range(len(mcc.Bmass))]
    mcjpsix.append(mcc)
    
mc = pd.concat(mc, ignore_index=True)
mcjpsix = pd.concat(mcjpsix, ignore_index=True)
data['w'] = np.ones(data.shape[0]) 
mc['w']   = 0.09 *1.1 *1.04 * 0.85 * 0.9 *1.4
mcjpsix['w'] = 0.3 * 0.85 *0.7*0.1 * 2.7 *1.6 *0.85 * 1.8 *1.4 * mcjpsix['jpsimother_weight']
data = pd.concat([data,mc,mcjpsix], ignore_index=True)
passing_tmp   = data.query('target == 1').copy()
failing_tmp   = data.query('target == 0').copy()

n_cuts = 40
aucs = {}

variables = histos.keys()


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
col = list(map(ROOT.TColor.GetColor, all_palettes['Set1'][9]))
z = 0

for j,var in enumerate(variables):
    if j%9==0 and j !=0:
        graph = ROOT.TGraph(n_cuts,xy, xy)
        #mg.Add(graph)
        mg.SetTitle("; Data(D) efficiency;MC(D) efficiency")
        mg.Draw("ac*")
        leg.Draw()

        c.SaveAs("ROC/plot_roc_alpha_"+str(int(j/11.))+".png")
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
        z=0
    print("Processing variable ",var)
    h_sig = ROOT.TH1F("h_sig_"+var,"h_sig_"+var,n_cuts,histos[var][3],histos[var][4])
    h_bkg = ROOT.TH1F("h_bkg_"+var,"h_bkg_"+var,n_cuts,histos[var][3],histos[var][4])

    for item in passing_tmp[var]:
        h_sig.Fill(item)
    for item in failing_tmp[var]:
        h_bkg.Fill(item)

    h_sig.SetMaximum(max(h_sig.GetMaximum(),h_bkg.GetMaximum()))
    h_sig.Draw("hist")
    h_bkg.SetLineColor(ROOT.kRed)
    h_bkg.Draw("sameHIST")
    c.SaveAs("ROC/plot_"+var+".png")
    print("signal integral",h_sig.Integral(),"bkg integral",h_bkg.Integral())

    eff_sig = array('d')
    rej_bkg = array('d')
    for i in range(n_cuts):
        point = i * (histos[var][4]-histos[var][3])/n_cuts + histos[var][3] 
        if(point == histos[var][3]): 
            eff_sig.append(0)
            rej_bkg.append(1.)
        else:
            eff_sig.append(h_sig.Integral(1,h_sig.GetXaxis().FindBin(point))/h_sig.Integral())
            eff_bkg = h_bkg.Integral(1,h_bkg.GetXaxis().FindBin(point))/h_bkg.Integral()
            rej_bkg.append(1 - eff_bkg)
    rej_bkg.append(0.)
    rej_bkg.append(1.)
    eff_sig.append(0.)
    eff_sig.append(0.)    
    #graph = ROOT.TGraph(n_cuts+2,rej_bkg,eff_sig)
    graph = ROOT.TGraph(n_cuts,rej_bkg,eff_sig)
    graph.SetTitle(var)
    print("col value",z)
    if z ==5:
        col[z] = ROOT.kGray
    graph.SetMarkerColor(col[z])
    graph.SetLineColor(col[z])
    mg.Add(graph)
    leg.AddEntry(graph,var,'L')
    xy.append(point)
    z+=1
    #aucs[var] = graph.Integral()-0.5
    
graph = ROOT.TGraph(n_cuts,xy, xy)
#mg.Add(graph)
mg.SetTitle("; Data(D) efficiency;MC(D) efficiency")
mg.Draw("ac*")
leg.Draw()

#c.BuildLegend()
c.SaveAs("ROC/plot_roc_alpha_"+str(int(j/11.)+1)+".png")
c.SaveAs("ROC/plot_roc_alpha_"+str(int(j/11.)+1)+".pdf")
#c.SaveAs("ROC/plot_allroc.png")

# sort aucs
sorted_aucs = sorted(aucs.items(), key=lambda kv: abs(kv[1]), reverse = True)
print(sorted_aucs)

for key in sorted_aucs:
    print(key)

final_array = [key[0] for key in sorted_aucs]
print(final_array)
