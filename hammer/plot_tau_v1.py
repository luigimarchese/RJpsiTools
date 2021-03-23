import ROOT
import os

ROOT.gStyle.SetOptStat(0)
ROOT.TH1.SetDefaultSumw2()
ROOT.gROOT.SetBatch()

#path_kis = 'flat_tree_kiselev_all_newgen.root'
#path_efg = 'reweighed_bc_tree_fromEfgtoKis.root'
path_kis = 'flat_tree_tau_kiselev.root'
path_efg = 'reweighed_bc_tree_tau_efgtokiselev2.root'

nbin = 12
xmin = -1
xmax = 12


branches_dic = {'pt_miss_vec':{'nbin' : nbin, 'xmin' : 0 , 'xmax' : 9},
                'q2':{'nbin' : nbin, 'xmin' : 4 , 'xmax' : 10},
                'e_star_mu3':{'nbin' : nbin, 'xmin' : 0.5 , 'xmax' : 2.2},
                'm2_miss':{'nbin' : nbin, 'xmin' : 0 , 'xmax' : 0.1},
                'pt_miss_sca':{'nbin' : nbin, 'xmin' : 0 , 'xmax' : 10}}

file_efg = ROOT.TFile(path_efg)
tree_efg = file_efg.Get("tree")

file_kis = ROOT.TFile(path_kis)
tree_kis = file_kis.Get("tree")

print(branches_dic)
for branch in branches_dic:
    print("Working on branch "+ branch)
    his_efg = ROOT.TH1F("his_efg","EFG",branches_dic[branch]['nbin'],branches_dic[branch]['xmin'],branches_dic[branch]['xmax'])
    his_rew_efg = ROOT.TH1F("his_rew_efg","EFG rew",branches_dic[branch]['nbin'],branches_dic[branch]['xmin'],branches_dic[branch]['xmax'])
    tree_efg.Draw(branch+">>his_efg","")

    for i in range(tree_efg.GetEntries()):
        tree_efg.GetEntry(i)
        his_rew_efg.Fill(getattr(tree_efg,branch),tree_efg.hammer)

    his_efg.Scale(1./his_efg.Integral())
    his_rew_efg.Scale(1./his_rew_efg.Integral())

    his_kis = ROOT.TH1F("his_kis","kiselev",branches_dic[branch]['nbin'],branches_dic[branch]['xmin'],branches_dic[branch]['xmax'])
    tree_kis.Draw(branch+">>his_kis","is_jpsi_tau & is3m & ismu3fromtau & bhad_pdgid == 541")
    his_kis.Scale(1./his_kis.Integral())

    maxx = max(his_efg.GetMaximum(),his_rew_efg.GetMaximum(),his_kis.GetMaximum())
    c = ROOT.TCanvas("","",700,700)
    c.Draw()
    his_efg.SetLineColor(ROOT.kRed)
    his_efg.SetMaximum(maxx*1.1)
    his_efg.GetYaxis().SetTitle("Normalized Events")
    his_efg.GetXaxis().SetTitle(branch)
    his_efg.Draw("e")
    his_kis.SetLineColor(ROOT.kBlack)
    his_kis.Draw("eSAME")
    his_rew_efg.SetLineColor(ROOT.kBlue)
    his_rew_efg.Draw("eSAME")
    c.Update()
    c.BuildLegend()
    c.SaveAs("hammer-validation_tau_"+branch+".png")

    
    his_ratio_efgkis = his_efg.Clone("his_ratio_efgkis")
    his_ratio_efgkis.Divide(his_efg,his_kis)
    his_ratio_efgrewkis = his_rew_efg.Clone("his_ratio_efgrewkis")
    his_ratio_efgrewkis.Divide(his_rew_efg,his_kis)
    his_ratio_kiskis = his_kis.Clone("his_ratio_kiskis")
    his_ratio_kiskis.Divide(his_kis,his_kis)
    maxx = max(his_ratio_efgkis.GetMaximum(),his_ratio_efgrewkis.GetMaximum(),his_ratio_kiskis.GetMaximum())
    print("Prob of EFG reweighted: ",his_ratio_efgrewkis.KolmogorovTest(his_ratio_kiskis))
    print("Prob of EFG: ",his_ratio_efgkis.KolmogorovTest(his_ratio_kiskis))
    #    print("p-value for  EFG reweighted: ",his_ratio_efgrewkis.Chi2Test(his_ratio_kiskis))
    #print("p-value for EFG: ",his_ratio_efgkis.Chi2Test(his_ratio_kiskis))
    #print(his_ratio_efgrewkis.GetBinContent(11))
    c1 = ROOT.TCanvas("","",700,700)
    c1.Draw()
    his_ratio_efgkis.SetLineColor(ROOT.kRed)
    his_ratio_efgkis.SetMaximum(maxx*1.1)
    #his_ratio_efgkis.SetMinimum(minn-0.1)
    his_ratio_efgkis.GetYaxis().SetTitle("Ratio with Kiselev Events")
    his_ratio_efgkis.Draw("e")
    his_ratio_kiskis.SetLineColor(ROOT.kBlack)
    his_ratio_kiskis.Draw("eSAME")
    his_ratio_efgrewkis.SetLineColor(ROOT.kBlue)
    his_ratio_efgrewkis.Draw("eSAME")
    c1.Update()
    c1.BuildLegend()
    c1.SaveAs("hammer-validation-ratio_tau_"+branch+".png")
    his_efg.Delete()
    his_rew_efg.Delete()
    his_kis.Delete()
    his_ratio_efgkis.Delete()
    his_ratio_kiskis.Delete()
    his_ratio_efgrewkis.Delete()
