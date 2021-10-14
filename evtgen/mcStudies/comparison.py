import os
import ROOT
import datetime
from glob import glob

ROOT.gStyle.SetOptStat(0)
ROOT.TH1.SetDefaultSumw2()
ROOT.gROOT.SetBatch()

f1 = ROOT.TFile.Open("RJpsi_HbToJPsiMuMu_3MuFilter_scale3_v7.root","r") # scaleToFactor3
f2 = ROOT.TFile.Open("/pnfs/psi.ch/cms/trivcat/store/user/friti/Rjpsi-HbToJpsiMuMu-3MuFilter_AOD_inspector_oldversion_scale1_v2/merged_file.root","r") # ScaleToFactor 1 (all produced)
f3 = ROOT.TFile.Open("RJpsi_HbToJPsiMuMu_3MuFilter_scale5_v6.root","r") # ScaleToFactor 5

tree_new_s3 = f1.Get("tree")
tree_new_s5 = f3.Get("tree")
tree_old = f2.Get("tree")

directory = 'comparison_plots/'

#create folder
if not os.path.exists(directory):
    os.makedirs(directory)
directory += datetime.date.today().strftime('%Y%b%d')+'/'
if not os.path.exists(directory):
    os.makedirs(directory)

c1= ROOT.TCanvas()

#defining muons
leading = "((mu1_pt)*(mu1_pt > mu2_pt & mu1_pt > mu3_pt) + (mu2_pt)*(mu2_pt > mu1_pt & mu2_pt > mu3_pt) + (mu3_pt)*(mu3_pt > mu2_pt & mu3_pt > mu1_pt))"
trailing = '((mu1_pt)*(mu1_pt < mu2_pt & mu1_pt < mu3_pt) + (mu2_pt)*(mu2_pt < mu1_pt & mu2_pt < mu3_pt) + (mu3_pt)*(mu3_pt < mu2_pt & mu3_pt < mu1_pt))'
subleading = '((mu1_pt)*(mu1_pt != '+leading+' & mu1_pt != '+trailing+') + (mu2_pt)*(mu2_pt != '+leading+' & mu2_pt != '+trailing+') + (mu3_pt)*(mu3_pt != '+leading+' & mu3_pt != '+trailing+'))'

#name; nbins; xmin; xmax
branches = {'mu1_pt': [20, 0, 30],
            'mu2_pt': [20, 0, 30],
            'mu3_pt': [20, 0, 30],
            #leading: [20, 0, 20],
            #trailing: [20, 0, 20],
            #subleading: [20, 0, 20],
            'mu1_eta': [20, -3, 3],
            'mu2_eta': [20, -3, 3],
            'mu3_eta': [20, -3 , 3],
            'mu1_phi': [20, -4, 4],
            'mu2_phi': [20, -4, 4],
            'mu3_phi': [20, -4, 4],
            'dr_jpsi_m': [20, 0, 1.],
            'dr12': [20, 0, 1],
            'dr23': [20, 0, 1],
            'dr13': [20, 0, 1],
            'm12': [20, 3.0, 3.15],
            'm23': [20, 0, 6],
            'm13': [20, 0, 6],
            'q2_reco': [25, -1, 13],
            'm2_miss_reco': [20, 0, 10],
            'e_star_mu3_reco': [20, 0, 5],
            'ptvar': [20, 0, 40]
}

selection = leading + " > 6 & "+subleading+" > 4 & "+trailing+">4 & abs(mu1_eta)<2.5 & abs(mu2_eta)<2.5 & abs(mu3_eta)<2.5 & mmm_m<6.3"

for branch in branches:
    h_new_s3 = ROOT.TH1F("h_new_s3","h_new_s3",branches[branch][0],branches[branch][1],branches[branch][2])
    h_new_s5 = ROOT.TH1F("h_new_s5","h_new_s5",branches[branch][0],branches[branch][1],branches[branch][2])
    h_old = ROOT.TH1F("h_old","h_old",branches[branch][0],branches[branch][1],branches[branch][2])

    h_new_s3.SetLineColor(ROOT.kRed)
    h_new_s5.SetLineColor(ROOT.kBlue)
    h_old.SetLineColor(ROOT.kBlack)

    print(branch)
    tree_new_s3.Draw(branch+">>h_new_s3",selection )
    tree_new_s5.Draw(branch+">>h_new_s5",selection )
    tree_old.Draw(branch+">>h_old",selection )

    h_new_s3.Scale(1./h_new_s3.Integral())    
    h_new_s5.Scale(1./h_new_s5.Integral())    
    h_old.Scale(1./h_old.Integral())

    # Plots normalized histos S3, S5 and old S1
    c1.Draw()
    maxx = max(h_new_s3.GetMaximum(),h_new_s5.GetMaximum(),h_old.GetMaximum())
    h_new_s3.Draw("histE")
    h_new_s3.SetMaximum(1.5*maxx)
    h_new_s3.SetTitle(";"+branch+";normalized events")
    h_new_s5.Draw("histE same")
    h_old.Draw("histE same")

    leg = ROOT.TLegend(0.54,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_new_s3, 'new S3', 'EP')
    leg.AddEntry(h_new_s5, 'new S5', 'EP')
    leg.AddEntry(h_old, 'old S1', 'EP')
    leg.Draw('same')
    
    ks = h_new_s3.KolmogorovTest(h_old)
    ks_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
    ks_value.AddText('ks S3 = %.2f' %ks)
    ks = h_new_s5.KolmogorovTest(h_old)

    ks_value.AddText('ks S5 = %.2f' %ks)
    ks_value.SetFillColor(0)
    ks_value.Draw('EP')
    c1.Update()

    c1.SaveAs(directory+'/'+branch+".png")


    # Plots all ratios
    c1.Draw()
    h_new_s5_ratio = h_new_s5.Clone("h_new_s5_ratio")
    h_new_s5_ratio.Divide(h_new_s5,h_old)
    h_new_s3_ratio = h_new_s3.Clone("h_new_s3_ratio")
    h_new_s3_ratio.Divide(h_new_s3,h_old)
    h_old_ratio = h_old.Clone("h_old_ratio")
    h_old_ratio.Divide(h_old,h_old)

    maxx = max(h_new_s5_ratio.GetMaximum(),h_new_s3_ratio.GetMaximum(),h_old_ratio.GetMaximum())
    h_new_s5_ratio.Draw("histE")
    h_new_s5_ratio.SetMaximum(1.5*maxx)
    h_new_s5_ratio.SetTitle("ratios;"+branch+";ratio")
    h_new_s3_ratio.Draw("histE same")
    h_old_ratio.Draw("histE same")
    

    leg = ROOT.TLegend(0.54,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_new_s5_ratio, 'new S5', 'EP')
    leg.AddEntry(h_new_s3_ratio, 'new S3', 'EP')
    leg.AddEntry(h_old_ratio, 'old', 'EP')
    leg.Draw('same')
    

    ks = h_new_s5_ratio.KolmogorovTest(h_old_ratio)
    ks2 = h_new_s3_ratio.KolmogorovTest(h_old_ratio)

    ks_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
    ks_value.AddText('ks_s5 = %.2f ' %(ks))
    ks_value.AddText('ks_s3 = %.2f' %(ks2))
    ks_value.SetFillColor(0)
    ks_value.Draw('EP')
    c1.Update()

    c1.Update()
    c1.SaveAs(directory+'/'+branch+"_ratio.png")
    
