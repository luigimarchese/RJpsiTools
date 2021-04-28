'''
Script to compare the shapes of data in the pass region with the shapes of MC HbX
I don't use RDataframes because I need to take the weights from the 2D plot of efficiency
and it's impossible
'''
import os
import ROOT
from datetime import datetime
from new_branches import to_define
from selections import preselection, pass_id, fail_id
from histos_nordf import histos #histos file no root dataframes
from cmsstyle import CMS_lumi
from officialStyle import officialStyle
from samples import weights, sample_names, titles
from root_pandas import read_root, to_root


ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

# Paths for Hb MC, data and the 2dplot of fakerate
hb_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_trigger_bcclean.root'
data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/data_ptmax_merged.root'
eff_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/plots_fr_eff/28Apr2021_18h14m13s/eff_2d.root'

def make_directories(label):
    '''
    Make directories for the plots
    '''
    os.system('mkdir -p plots_passfail_weights/%s/pdf/lin/' %label)
    os.system('mkdir -p plots_passfail_weights/%s/png/lin/' %label)

def save_selection(label, preselection,pass_id,fail_id):
    '''
    Save preselections in a txt file
    '''
    with open('plots_pf_comparison/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '):
            print("    '%s'," %isel, file=ff)
        
        print('])', file=ff)
        print("pass_id = '%s'"%pass_id,file=ff)
        print("fail_id = '%s'"%fail_id,file=ff)

# Upload trees ad dataframes
tree_name = 'BTo3Mu'
hb= read_root(hb_path, 'BTo3Mu', where=preselection + '& !(abs(k_genpdgId)==13)')
data= read_root(data_path, 'BTo3Mu', where=preselection)
fin = ROOT.TFile(eff_path,"r")
eff2d= fin.Get("eff")

print("##########################")
print("##### Samples loaded #####")
print("##########################")

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
make_directories(label)

#Define pass and fail regions both for mc and data
hb_pass = hb[hb.k_mediumID >0.5]
hb_fail = hb[hb.k_mediumID <=0.5]
data_pass = data[data.k_mediumID >0.5]
data_fail = data[data.k_mediumID <=0.5]

#Define canvas
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
main_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.2)

# for each variable we do this comparison
for var in histos:
        print("Computing now variable "+ var)
        hist_pass = ROOT.TH1D("pass"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in hb_pass[histos[var][0]]:
            hist_pass.Fill(item)
        hist_fail = ROOT.TH1D("fail"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in hb_fail[histos[var][0]]:
            hist_fail.Fill(item)
        hist_data_fail = ROOT.TH1D("faild"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in data_fail[histos[var][0]]:
            hist_data_fail.Fill(item)
        # The weight is taken from the eff2d tefficiency histogram
        hist_data_pass = ROOT.TH1D("passd"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item,pt,eta in zip(data_pass[histos[var][0]],data_pass['kpt'],data_pass['keta']):
            binx = eff2d.GetXaxis().FindBin(pt)
            biny = eff2d.GetYaxis().FindBin(eta)
            weight = eff2d.GetBinContent(binx,biny)
            hist_data_pass.Fill(item,weight)


        c1.cd()
        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        main_pad.cd()
        main_pad.SetLogy(False)

        hist_pass.GetXaxis().SetTitle(histos[var][5])
        hist_pass.GetYaxis().SetTitle('events')
        hist_pass.SetLineColor(ROOT.kMagenta)
        hist_pass.SetFillColor(0)

        hist_fail.GetXaxis().SetTitle(histos[var][5])
        hist_fail.GetYaxis().SetTitle('events')
        hist_fail.SetLineColor(ROOT.kBlue)
        hist_fail.SetFillColor(0)

        hist_data_fail.SetLineColor(ROOT.kGreen)
        hist_data_fail.SetFillColor(0)
        hist_data_pass.SetLineColor(ROOT.kRed)
        hist_data_pass.SetFillColor(0)

        hist_pass.Scale(1./hist_pass.Integral())
        hist_fail.Scale(1./hist_fail.Integral())
        hist_data_pass.Scale(1./hist_data_pass.Integral())
        hist_data_fail.Scale(1./hist_data_fail.Integral())
        maximum = max(hist_pass.GetMaximum(),hist_fail.GetMaximum(),hist_data_pass.GetMaximum(),hist_data_pass.GetMaximum()) 
        hist_pass.SetMaximum(2.*maximum)
        #        hist_fail.Draw('hist')
        hist_pass.Draw('hist ')
        hist_data_pass.Draw('hist same')
        hist_data_fail.Draw('hist same')

        #        leg.AddEntry(hist_fail, 'hb_fail','F')
        leg.AddEntry(hist_pass, 'hb_pass','F')
        leg.AddEntry(hist_data_fail, 'data_fail','F')
        leg.AddEntry(hist_data_pass, 'data_pass','F')

        leg.Draw('same')
    
        CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

        main_pad.cd()
    
        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_passfail_weights/%s/pdf/lin/%s.pdf' %(label, var))
        c1.SaveAs('plots_passfail_weights/%s/png/lin/%s.png' %(label, var))
    
save_selection(label, preselection,pass_id,fail_id)
