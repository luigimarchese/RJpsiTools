'''
Script to plot the up/central and down/central of ctau for mu
'''
import ROOT

ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)


fin = ROOT.TFile.Open('plots_ul/03May2021_15h16m29s/datacards/datacard_pass_decay_time_ps.root','r')
sname = 'jpsi_mu'
syst = 'ctau'

maxx = []

c2 = ROOT.TCanvas('c2', '', 700, 700)
c2.Draw()
c2.cd()
c2.SetTicks(True)
c2.SetBottomMargin(0.15)

# central histo
his_central = fin.Get(sname)
xmin = his_central.GetBinLowEdge(1)
xmax = his_central.GetBinLowEdge(his_central.GetNbinsX() + 1)
nbins = his_central.GetNbinsX()            
histo_central = ROOT.TH1F(sname,sname, nbins, xmin, xmax)
for i in range(1,his_central.GetNbinsX()+1):
    histo_central.SetBinContent(i,his_central.GetBinContent(i))
    histo_central.SetBinError(i,his_central.GetBinError(i))

#Up histo
his_up = fin.Get(sname+'_'+syst+'Up')
histo_up = ROOT.TH1F(sname+'_'+syst+'Up',sname+'_'+syst+'Up', nbins, xmin, xmax)
for i in range(1,his_up.GetNbinsX()+1):
    histo_up.SetBinContent(i,his_up.GetBinContent(i))
    histo_up.SetBinError(i,his_up.GetBinError(i))

#down histo
his_down = fin.Get(sname+'_'+syst+'Down')
histo_down = ROOT.TH1F(sname+'_'+syst+'Down',sname+'_'+syst+'Down', nbins, xmin, xmax)
for i in range(1,his_down.GetNbinsX()+1):
    histo_down.SetBinContent(i,his_down.GetBinContent(i))
    histo_down.SetBinError(i,his_down.GetBinError(i))

histo_ratio_up = histo_up.Clone(sname+'_'+syst+'Up')
histo_ratio_up.Divide(histo_up,histo_central)

histo_ratio_down = histo_down.Clone(sname+'_'+syst+'Down')
histo_ratio_down.Divide(histo_down,histo_central)

maxx.append(histo_ratio_up.GetMaximum())
maxx.append(histo_ratio_down.GetMaximum())


histo_ratio_up.SetTitle(sname+' '+syst+';t (ps) ;events')
histo_ratio_up.SetLineColor(ROOT.kBlack)
histo_central.SetMarkerStyle(8)
histo_ratio_up.SetMarkerColor(ROOT.kRed)
histo_ratio_up.SetMarkerStyle(8)
histo_ratio_down.SetMarkerColor(ROOT.kGreen)
histo_ratio_down.SetMarkerStyle(8)
histo_ratio_up.SetMaximum(1.2)
histo_ratio_up.SetMinimum(0.8)
histo_ratio_up.Draw("hist p")
histo_ratio_down.Draw("hist p same")

leg = ROOT.TLegend(0.5,.75,.95,.90)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)
leg.AddEntry(histo_ratio_up, 'up value/central',  'p')
leg.AddEntry(histo_ratio_down, 'down value/central',  'p')
leg.Draw('same')

#CMS_lumi(c2, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

c2.Modified()
c2.Update()

c2.SaveAs('plots_ul/03May2021_15h16m29s/datacards/ctau_up_down_comparison_jpsimu.png')
