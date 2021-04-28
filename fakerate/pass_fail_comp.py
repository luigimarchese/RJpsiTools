import os
import ROOT
from datetime import datetime
from new_branches import to_define
from selections import preselection, pass_id, fail_id
from histos import histos
from cmsstyle import CMS_lumi
from officialStyle import officialStyle
from samples import weights, sample_names, titles

# preselection
# muonID

#Only need data

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/'
tree_data = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/'
def make_directories(label):

    os.system('mkdir -p plots_pf_comparison/%s/pdf/lin/' %label)
    os.system('mkdir -p plots_pf_comparison/%s/png/lin/' %label)

def save_selection(label, preselection,pass_id,fail_id):
    with open('plots_pf_comparison/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '):
            print("    '%s'," %isel, file=ff)
        
        print('])', file=ff)
        print("pass_id = '%s'"%pass_id,file=ff)
        print("fail_id = '%s'"%fail_id,file=ff)
        
tree_name = 'BTo3Mu'
hb= ROOT.RDataFrame(tree_name,'%s/HbToJPsiMuMu_ptmax_merged.root' %(tree_dir))
data = ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged.root' %(tree_data))

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

# create plot directories
make_directories(label)

c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
main_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.2)

#define jpsiK_mass
for new_column, new_definition in to_define:
    if hb.HasColumn(new_column):
        continue
    hb = hb.Define(new_column, new_definition)
for new_column, new_definition in to_define:
    if data.HasColumn(new_column):
        continue
    data = data.Define(new_column, new_definition)

hb = hb.Filter(preselection) #NOT preselection_mc
data = data.Filter(preselection) #NOT preselection_mc

# for hb I want only NOT real muons as third muon
hb = hb.Filter('!(abs(k_genpdgId)==13)')

temp_hists      = {} # pass muon ID category
temp_hists_fake = {} # fail muon ID category
temp_hists_data = {} # fail muon ID category

for k, v in histos.items():
        temp_hists     [k] = hb.Filter(pass_id).Histo1D(v[0], k)
        temp_hists_fake[k] = hb.Filter(fail_id).Histo1D(v[0], k)
        temp_hists_data[k] = data.Filter(fail_id).Histo1D(v[0],k)
print(hb.Filter(fail_id).Count().GetValue())
'''

for k, v in histos.items():
        temp_hists     [k] = hb.Filter(pass_condition).Histo1D(v[0], k)
        temp_hists_fake[k] = hb.Filter(fail_condition).Histo1D(v[0], k)
'''

print('====> now looping')
# loop on the histos
for k, v in histos.items():

    c1.cd()
    leg = ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    main_pad.cd()
    main_pad.SetLogy(False)

    key = k
    ihist_pass = temp_hists[k] 
    ihist_fail = temp_hists_fake[k] 
    ihist_data = temp_hists_data[k]

    ihist_pass.GetXaxis().SetTitle(k)
    ihist_pass.GetYaxis().SetTitle('events')
    ihist_pass.SetLineColor(ROOT.kMagenta)
    ihist_pass.SetFillColor(0)

    ihist_fail.GetXaxis().SetTitle(k)
    ihist_fail.GetYaxis().SetTitle('events')
    ihist_fail.SetLineColor(ROOT.kBlue)
    ihist_fail.SetFillColor(0)

    ihist_data.SetLineColor(ROOT.kGreen)
    ihist_data.SetFillColor(0)

    ihist_pass.Scale(1./ihist_pass.Integral())
    ihist_fail.Scale(1./ihist_fail.Integral())
    ihist_data.Scale(1./ihist_data.Integral())
    maximum = max(ihist_pass.GetMaximum(),ihist_fail.GetMaximum(),ihist_data.GetMaximum()) 
    ihist_pass.SetMaximum(2.*maximum)
    ihist_pass.Draw('hist')
    ihist_fail.Draw('hist same')
    ihist_data.Draw('hist same')

    leg.AddEntry(ihist_pass.GetValue(), 'hb_pass','F')
    leg.AddEntry(ihist_fail.GetValue(), 'hb_fail','F')
    leg.AddEntry(ihist_data.GetValue(), 'data_fail','F')

    leg.Draw('same')
    
    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    main_pad.cd()
    
    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_pf_comparison/%s/pdf/lin/%s.pdf' %(label, k))
    c1.SaveAs('plots_pf_comparison/%s/png/lin/%s.png' %(label, k))
    
save_selection(label, preselection,pass_id,fail_id)
