'''
Script to compare the shapes of data in the pass region with the shapes of MC HbX
I don't use RDataframes because I need to take the weights from the 2D plot of efficiency
and it's impossible
'''
import os
import ROOT
from datetime import datetime
from cmsstyle import CMS_lumi
from officialStyle import officialStyle
from root_pandas import read_root, to_root
import time

from new_branches import to_define
from selections import preselection, pass_id, fail_id
from histos_nordf import histos #histos file NO root dataframes
from samples import weights, sample_names, titles

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)

officialStyle(ROOT.gStyle, ROOT.TGaxis)

efficiency_folder = '30Apr2021_09h31m39s'
load_data = True

# Paths for Hb MC, data and the 2dplot of fakerate
hb_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/HbToJPsiMuMu_trigger_bcclean.root'
data_path = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15/data_ptmax_merged.root'
pteta_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/plots_fr_eff/'+efficiency_folder+'/eff_2d.root'
dr_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/plots_fr_eff/'+efficiency_folder+'/dr.root'
mass_path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/fakerate/plots_fr_eff/'+efficiency_folder+'/bmass.root'

def make_directories(label):
    '''
    Make directories for the plots
    '''
    os.system('mkdir -p plots_closuretest/%s/pteta/pdf/' %label)
    os.system('mkdir -p plots_closuretest/%s/pteta/png/' %label)
    os.system('mkdir -p plots_closuretest/%s/dr/pdf/' %label)
    os.system('mkdir -p plots_closuretest/%s/dr/png/' %label)
    os.system('mkdir -p plots_closuretest/%s/mass/pdf/' %label)
    os.system('mkdir -p plots_closuretest/%s/mass/png/' %label)

def save_selection(label, preselection,pass_id,fail_id):
    '''
    Save preselections in a txt file
    '''
    with open('plots_closuretest/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '):
            print("    '%s'," %isel, file=ff)
        
        print('])', file=ff)
        print("pass_id = '%s'"%pass_id,file=ff)
        print("fail_id = '%s'"%fail_id,file=ff)

# Upload trees ad dataframes
tree_name = 'BTo3Mu'
#hb = read_root(hb_path, 'BTo3Mu', where=preselection + '& !(abs(k_genpdgId)==13)')
hb_pass = read_root(hb_path, 'BTo3Mu', where=preselection + '&' + pass_id + '& !(abs(k_genpdgId)==13)')
hb_fail = read_root(hb_path, 'BTo3Mu', where=preselection + '&' + fail_id + '& !(abs(k_genpdgId)==13)')

if load_data:
    data_fail= read_root("data_fail_30Apr.root", 'BTo3Mu', where=preselection + '&' + fail_id)
else:
    data_fail= read_root(data_path, 'BTo3Mu', where=preselection + '&' + fail_id)
    data_fail.to_root("data_fail_30Apr.root",key = 'BTo3Mu')

fin_pteta = ROOT.TFile(pteta_path,"r")
fin_dr = ROOT.TFile(dr_path,"r")
fin_mass = ROOT.TFile(mass_path,"r")
eff_pteta= fin_pteta.Get("eff")
eff_dr= fin_dr.Get("eff")
eff_mass= fin_mass.Get("eff")

print("#####################################")
print("##### Samples and histos loaded #####")
print("#####################################")

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
make_directories(label)

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
        
        #histo for the MC in the pass region
        hist_pass = ROOT.TH1D("pass"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in hb_pass[histos[var][0]]:
            hist_pass.Fill(item)

        #histo from the fail region reweighted to be in the pass
        hist_pass_wpteta = ROOT.TH1D("passw"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item,pt,eta in zip(hb_fail[histos[var][0]],hb_fail['kpt'],hb_fail['keta']):
            binx_pteta = eff_pteta.GetXaxis().FindBin(pt)
            biny_pteta = eff_pteta.GetYaxis().FindBin(abs(eta))
            weight_pteta = eff_pteta.GetBinContent(binx_pteta,biny_pteta)
            if weight_pteta == 1:
                hist_pass_wpteta.Fill(item)
            else:
                hist_pass_wpteta.Fill(item,weight_pteta/(1-weight_pteta))

        hist_pass_wdr = ROOT.TH1D("passwdr"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item,deltar in zip(hb_fail[histos[var][0]],hb_fail['DR_mu1mu2']):
            binx_dr = eff_dr.GetXaxis().FindBin(deltar)
            weight_dr = eff_dr.GetBinContent(binx_dr)
            if weight_dr == 1:
                hist_pass_wdr.Fill(item)
            else:
                hist_pass_wdr.Fill(item,weight_dr/(1-weight_dr))

        hist_pass_wmass = ROOT.TH1D("passwmass"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item,mass in zip(hb_fail[histos[var][0]],hb_fail['Bmass']):
            binx_mass = eff_mass.GetXaxis().FindBin(mass)
            weight_mass = eff_mass.GetBinContent(binx_mass)
            if weight_mass == 1:
                hist_pass_wmass.Fill(item)
            else:
                hist_pass_wmass.Fill(item,weight_mass/(1-weight_mass))

        
        # histo for the MC in the fail reigon
        hist_fail = ROOT.TH1D("fail"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in hb_fail[histos[var][0]]:
            hist_fail.Fill(item)

        # Histo for data in the fail region
        hist_data_fail = ROOT.TH1D("faild"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        for item in data_fail[histos[var][0]]:
            hist_data_fail.Fill(item)

        # histo of fakes taken from the fail region of data into the pass region -> using weights
        # The weight for the pass region of data is taken from the eff2d tefficiency histogram
        hist_fakes_pass = ROOT.TH1D("passd"+histos[var][0],"",histos[var][2],histos[var][3],histos[var][4])
        '''for item,pt,eta in zip(data_fail[histos[var][0]],data_fail['kpt'],data_fail['keta']):
            binx = eff2d.GetXaxis().FindBin(pt)
            biny = eff2d.GetYaxis().FindBin(abs(eta))
            weight = eff2d.GetBinContent(binx,biny)
            #print(weight)
            #print(pt,binx,eta,biny,weight)
            if weight == 1:
                hist_fakes_pass.Fill(item)
            else:
                hist_fakes_pass.Fill(item,weight/(1-weight))
        '''

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

        hist_pass_wpteta.GetXaxis().SetTitle(histos[var][5])
        hist_pass_wpteta.GetYaxis().SetTitle('events')
        hist_pass_wpteta.SetLineColor(ROOT.kOrange)
        hist_pass_wpteta.SetFillColor(0)

        hist_pass_wdr.GetXaxis().SetTitle(histos[var][5])
        hist_pass_wdr.GetYaxis().SetTitle('events')
        hist_pass_wdr.SetLineColor(ROOT.kOrange)
        hist_pass_wdr.SetFillColor(0)

        hist_pass_wmass.GetXaxis().SetTitle(histos[var][5])
        hist_pass_wmass.GetYaxis().SetTitle('events')
        hist_pass_wmass.SetLineColor(ROOT.kOrange)
        hist_pass_wmass.SetFillColor(0)

        hist_fail.GetXaxis().SetTitle(histos[var][5])
        hist_fail.GetYaxis().SetTitle('events')
        hist_fail.SetLineColor(ROOT.kBlue)
        hist_fail.SetFillColor(0)

        hist_data_fail.SetLineColor(ROOT.kGreen)
        hist_data_fail.SetFillColor(0)

        hist_fakes_pass.SetLineColor(ROOT.kRed)
        hist_fakes_pass.SetFillColor(0)

        hist_pass.Scale(1./hist_pass.Integral())
        hist_pass_wpteta.Scale(1./hist_pass_wpteta.Integral())
        hist_pass_wdr.Scale(1./hist_pass_wdr.Integral())
        hist_pass_wmass.Scale(1./hist_pass_wmass.Integral())
        hist_fail.Scale(1./hist_fail.Integral())
        #hist_fakes_pass.Scale(1./hist_fakes_pass.Integral())
        #hist_data_fail.Scale(1./hist_data_fail.Integral())
        maximum = max(hist_pass.GetMaximum(),hist_fail.GetMaximum(),hist_fakes_pass.GetMaximum(),hist_fakes_pass.GetMaximum(),hist_pass_wpteta.GetMaximum(),hist_pass_wdr.GetMaximum(),hist_pass_wmass.GetMaximum()) 

        hist_pass.SetMaximum(2.*maximum)

        CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

        #########################    
        ####### pt eta ##########
        #########################    

        hist_pass.Draw('hist ')
        hist_fail.Draw('hist same')
        hist_pass_wpteta.Draw('hist same')

        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        leg.AddEntry(hist_pass, 'hb_pass','F')
        leg.AddEntry(hist_fail, 'hb_fail','F')
        leg.AddEntry(hist_pass_wpteta, 'hb_fail_pteta','F')

        leg.Draw('same')
    
        main_pad.cd()
    
        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_closuretest/%s/pteta/pdf/%s.pdf' %(label, var))
        c1.SaveAs('plots_closuretest/%s/pteta/png/%s.png' %(label, var))

        #########################    
        ######### dr ############
        #########################    

        hist_pass.Draw('hist ')
        hist_fail.Draw('hist same')
        hist_pass_wdr.Draw('hist same')

        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        leg.AddEntry(hist_pass, 'hb_pass','F')
        leg.AddEntry(hist_fail, 'hb_fail','F')
        leg.AddEntry(hist_pass_wdr, 'hb_fail_dr','F')

        leg.Draw('same')

        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_closuretest/%s/dr/pdf/%s.pdf' %(label, var))
        c1.SaveAs('plots_closuretest/%s/dr/png/%s.png' %(label, var))

        #########################    
        ######## mass ###########
        #########################    

        hist_pass.Draw('hist ')
        hist_fail.Draw('hist same')
        hist_pass_wmass.Draw('hist same')

        leg = ROOT.TLegend(0.24,.67,.95,.90)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)

        leg.AddEntry(hist_pass, 'hb_pass','F')
        leg.AddEntry(hist_fail, 'hb_fail','F')
        leg.AddEntry(hist_pass_wmass, 'hb_fail_mass','F')

        leg.Draw('same')

        c1.Modified()
        c1.Update()
        c1.SaveAs('plots_closuretest/%s/mass/pdf/%s.pdf' %(label, var))
        c1.SaveAs('plots_closuretest/%s/mass/png/%s.png' %(label, var))
        
save_selection(label, preselection,pass_id,fail_id)
