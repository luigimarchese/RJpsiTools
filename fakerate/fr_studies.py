'''
This script studies the dependency of the fake rate
with other variables as the pt or the eta of the muon
'''
from root_pandas import read_root, to_root
from datetime import datetime
from cmsstyle import CMS_lumi
from officialStyle import officialStyle
import ROOT
import os

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
variables = {'kpt':
             {'nbins':50, 'xmax':25},
             'keta':
             {'nbins':50, 'xmax':2.5}}

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

# create plot directories
make_directories(label)
tree_pass = tree_df[tree_df.k_mediumID>0.5]
# histo of pt pass and fail
for var in variables:
    his_pass = ROOT.TH1F("pass","pass",variables[var]['nbins'],0,variables[var]['xmax'])
    his_tot = ROOT.TH1F("tot","tot",variables[var]['nbins'],0,variables[var]['xmax'])

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

# 2d fakerate
histo_2d_pass = ROOT.TH2F("pass2d","pass2d",20,0,variables['kpt']['xmax'],20,0,variables['keta']['xmax'])
histo_2d_tot = ROOT.TH2F("tot2d","tot2d",20,0,variables['kpt']['xmax'],20,0,variables['keta']['xmax'])
eff_2d = ROOT.TH2F("eff","eff",20,0,variables['kpt']['xmax'],20,0,variables['keta']['xmax'])

for pt,eta in zip(tree_pass['kpt'],tree_pass['keta']):
    histo_2d_pass.Fill(pt,eta)
for pt,eta in zip(tree_df['kpt'],tree_df['keta']):
    histo_2d_tot.Fill(pt,eta)

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

fout = ROOT.TFile.Open('plots_fr_eff/%s/eff_2d.root' %(label), 'recreate')
fout.cd()
eff_2d.Write()
fout.Close()
