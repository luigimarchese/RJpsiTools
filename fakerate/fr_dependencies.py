'''
This script studies the dependency of the fake rate
to variables as the pt or the eta of the muon
'''
from root_pandas import read_root, to_root
from datetime import datetime
import ROOT
import os
from array import array

from cmsstyle import CMS_lumi
from officialStyle import officialStyle

from selections import preselection, pass_id, fail_id

#no pop-up windows
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
officialStyle(ROOT.gStyle, ROOT.TGaxis)

def make_directories(label):
    ''' make plot directories'''
    os.system('mkdir -p plots_fr_eff/%s/pdf/' %label)
    os.system('mkdir -p plots_fr_eff/%s/png/' %label)

#Open the Hb->Jpsi(mumu) X MC
hb_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_trigger_bcclean.root' #Hb

# Ask that the third particle is not a real muon and for the preselection
tree_df = read_root(hb_path, 'BTo3Mu', where=preselection + '& !(abs(k_genpdgId)==13)')
tree_pass = read_root(hb_path, 'BTo3Mu', where=preselection + '&'+pass_id+'& !(abs(k_genpdgId)==13)')

variables = {'kpt':
             {'nbins':100, 'xmin':2,'xmax':25},
             'keta':
             {'nbins':50,'xmin':-2.5, 'xmax':2.5},
             'DR_mu1mu2':
             {'nbins':10,'xmin':0, 'xmax':1},
             'Bmass':
             {'nbins':10,'xmin':0, 'xmax':6.3}}

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
make_directories(label)

#MC in the pass region
#tree_pass = tree_df[(tree_df.k_mediumID>0.5) & (tree_df.k_raw_db_corr_iso03_rel<0.2)].copy()
# 1D histos of fr vs pt and fr vs eta
for var in variables:
    his_pass = ROOT.TH1F("pass","pass",variables[var]['nbins'],variables[var]['xmin'],variables[var]['xmax'])
    his_tot = ROOT.TH1F("tot","tot",variables[var]['nbins'],variables[var]['xmin'],variables[var]['xmax'])

    for item in tree_pass[var]:
        his_pass.Fill(item)
    for item in tree_df[var]:
        his_tot.Fill(item)

    eff = ROOT.TEfficiency(his_pass,his_tot)
    eff.SetTitle(';'+var+'; eff')
    
    c1 = ROOT.TCanvas('c1', '', 700, 700)
    c1.Draw()
    c1.cd()
    
    eff.Draw()
    
    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
    
    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_fr_eff/%s/pdf/%s.pdf' %(label, var))
    c1.SaveAs('plots_fr_eff/%s/png/%s.png' %(label, var))

    if var == 'DR_mu1mu2':
        #saves the root file
        foutdr = ROOT.TFile.Open('plots_fr_eff/%s/dr.root' %(label), 'recreate')
        foutdr.cd()
        effPt = his_tot.Clone("eff");
        effPt.Divide(his_pass, his_tot);
        effPt.Write()
        foutdr.Close()

    if var == 'Bmass':
        #saves the root file
        foutdr = ROOT.TFile.Open('plots_fr_eff/%s/bmass.root' %(label), 'recreate')
        foutdr.cd()
        effPt = his_tot.Clone("eff");
        effPt.Divide(his_pass, his_tot);
        effPt.Write()
        foutdr.Close()

    if var == 'kpt':
        y_edges = [2.]
        total_events = his_tot.Integral()
        events_in1bin = total_events/8.
        #print(total_events,events_in1bin)
        count = 0.
        for i in range(1,his_tot.GetNbinsX()+1):
            count += his_tot.GetBinContent(i)
            if count >= events_in1bin:
                y_edges.append(his_tot.GetBinLowEdge(i+1))
                print(count)
                count = 0.
        y_edges.append(25.)
print(y_edges)

# 2d fakerate histo

pt_histo_edges = array('d',y_edges)
histo_2d_pass = ROOT.TH2F("pass2d","pass2d",8,pt_histo_edges,5,0,variables['keta']['xmax'])
histo_2d_tot = ROOT.TH2F("tot2d","tot2d",8,pt_histo_edges,5,0,variables['keta']['xmax'])
eff_2d = ROOT.TH2F("eff","eff",8,pt_histo_edges,5,0,variables['keta']['xmax'])

for pt,eta in zip(tree_pass['kpt'],tree_pass['keta']):
    histo_2d_pass.Fill(pt,abs(eta))
for pt,eta in zip(tree_df['kpt'],tree_df['keta']):
    histo_2d_tot.Fill(pt,abs(eta))

#eff_2d = ROOT.TEfficiency(histo_2d_pass,histo_2d_tot)
eff_2d.Divide(histo_2d_pass,histo_2d_tot)
eff_2d.SetTitle(';p_{T}^{#mu}; #eta^{#mu}; eff')
    
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()

eff_2d.Draw("colz")

CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

c1.Modified()
c1.Update()
c1.SaveAs('plots_fr_eff/%s/pdf/eff_2d.pdf' %(label))
c1.SaveAs('plots_fr_eff/%s/png/eff_2d.png' %(label))

#saves the root file
fout = ROOT.TFile.Open('plots_fr_eff/%s/eff_2d.root' %(label), 'recreate')
fout.cd()
eff_2d.Write()
fout.Close()
