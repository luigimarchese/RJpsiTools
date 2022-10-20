import ROOT
from root_pandas import read_root, to_root
from selections_for_fakerate import preprepreselection,triggerselection, etaselection
from new_branches_pandas import to_define
from array import array
from histos_nordf import histos
from bokeh.palettes import viridis, all_palettes
import pandas as pd


ROOT.gStyle.SetOptStat(0)

id_variable = 'k_softMvaId'

print(histos.keys())

# data
data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_June2022/data_nopresel_withpresel_v2_withnn_withidiso.root'
prepreselection = preprepreselection + "&" +triggerselection +"&"+etaselection + "& Bpt_reco<80"
#preselection and not-true muon request
data = read_root(data_path, 'BTo3Mu', where=prepreselection )
#data.index = np.array(range(len(data)))
data = to_define(data)
main_df = data

#check in the fail ID region which features are better for the training of the NN
pass_id = id_variable+'<0.5 & k_raw_db_corr_iso03_rel<0.2' 
fail_id = id_variable+'<0.5 & (k_raw_db_corr_iso03_rel>0.2)'

#main_df is already shuffled
passing_tmp   = main_df.query(pass_id).copy()
failing_tmp   = main_df.query(fail_id).copy()

n_cuts = 40

#variables = histos.keys()
variables = histos.keys()
#variables = ['Bmass']
#variables = ['m_miss_sq','jpsivtx_lxy_unc','jpsivtx_cos2D','Q_sq','pt_var','pt_miss_vec','pt_miss_scal','E_mu_star','E_mu_canc','DR_mu1mu2','jpsi_pt','Bmass','mu1pt','mu2pt','kpt','Bpt','Bpt_reco','bvtx_lxy_unc','dr12','dr23','dr13','jpsiK_mass']

aucs = {}

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
        mg.SetTitle("; Data(D) efficiency;Data(C) efficiency")
        mg.Draw("ac*")
        leg.Draw()
        c.SaveAs("ROC/plot_roc_data_"+str(int(j/11.))+".png")
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
    if z == 5:
        col[z] = ROOT.kGray
    graph.SetMarkerColor(col[z])
    graph.SetLineColor(col[z])
    mg.Add(graph)
    leg.AddEntry(graph,var,'L')
    xy.append(point)
    z+=1

    #aucs[var] = graph.Integral()-0.5
graph = ROOT.TGraph(n_cuts,xy, xy)

mg.SetTitle("; Data(D) efficiency;Data(C) efficiency")
mg.Draw("ac*")
leg.Draw()

#c.BuildLegend()
c.SaveAs("ROC/plot_roc_data_"+str(int(j/11.)+1)+".png")
c.SaveAs("ROC/plot_roc_data_"+str(int(j/11.)+1)+".pdf")

# sort aucs
sorted_aucs = sorted(aucs.items(), key=lambda kv: abs(kv[1]), reverse = True)
print(sorted_aucs)

for key in sorted_aucs:
    print(key)

final_array = [key[0] for key in sorted_aucs]
print(final_array)
