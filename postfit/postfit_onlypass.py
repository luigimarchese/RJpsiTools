from cmsstyle import CMS_lumi
from bokeh.palettes import viridis, all_palettes
import ROOT
from datetime import datetime
import os
from samples_wf import weights, sample_names, titles
from officialStyle import officialStyle
from histos import histos

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

#### var info ####
var = 'Q_sq'
cut = '1'

#### fit path infos ####
fit_date = "31Mar2021_15h35m38s"
path_comb = '/work/friti/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'

#output
label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
os.system('mkdir -p plots_postfit/%s/pdf/'%label)
os.system('mkdir -p plots_postfit/%s/png/'%label)
os.system('mkdir -p plots_postfit/%s/inputs/'%label)

#inputs
#copy datacard; root files
os.system("cp "+ path_comb + var+"_"+fit_date+"_cut"+cut+ "/datacard.txt " + "plots_postfit/"+label+"/inputs/.")
os.system("cp "+ path_comb +var+"_"+fit_date+"_cut"+cut+ "/datacard_pass_"+var+".root plots_postfit/"+label+"/inputs/.")
os.system("cp "+ path_comb +var+"_"+fit_date+"_cut"+cut+ "/fitDiagnostics.root plots_postfit/"+label+"/inputs/.")

#symbolic link to original plot folder
symb_out = os.popen("find "+path_comb + var+"_"+fit_date+"_cut"+cut +" . -maxdepth 1 -type l -ls").readlines()[0]
splitted1 = symb_out.split(" ")
os.system("ln -s "+ splitted1[-3]+" plots_postfit/"+label+"/inputs/")

#take xmin and xmax automatically from histo
f=ROOT.TFile("plots_postfit/"+label+"/inputs/datacard_pass_"+var+".root","r")
his = f.Get("jpsi_tau")
xmin = his.GetBinLowEdge(1)
xmax = his.GetBinLowEdge(his.GetNbinsX() + 1)
nbins = his.GetNbinsX()
f.Close()

officialStyle(ROOT.gStyle, ROOT.TGaxis)

f=ROOT.TFile("plots_postfit/"+label+"/inputs/fitDiagnostics.root","r")
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0.25, 1. , 1.  )
main_pad.Draw()
c1.cd()
ratio_pad = ROOT.TPad('ratio_pad', '', 0., 0., 1., 0.25)
ratio_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.)
# main_pad.SetTopMargin(0.3)   
# main_pad.SetLeftMargin(0.15)
# main_pad.SetRightMargin(0.15)
# ratio_pad.SetLeftMargin(0.15)
# ratio_pad.SetRightMargin(0.15)
ratio_pad.SetTopMargin(0.)   
ratio_pad.SetGridy()
ratio_pad.SetBottomMargin(0.45)

'''
sample_names_new = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
    'onia'     ,
#     'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
    #    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
    #'data'     ,
    'fakes'
]
'''
colours = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][len(sample_names)-1]))

for channel in ['signal']:
    ths1      = ROOT.THStack('stack', '')
    c1.cd()

    main_pad.cd()
    main_pad.SetLogy(False)

    maxima = []
    data_max = 0.
    hs = []
    for j,iname in enumerate(sample_names[:-1]): # except data
        histo = f.Get("shapes_fit_s/" + channel + "/" + iname)
        histo_new = ROOT.TH1F(iname,iname, nbins, xmin, xmax)

        try:
            for i in range(1,histo.GetNbinsX()+1):
                histo_new.SetBinContent(i,histo.GetBinContent(i))
                histo_new.SetBinError(i,histo.GetBinError(i))
                #histo_new.Scale(scale)
        except: #if histo is empy from before the fit, combine doesn't save it at all in fitdiagnostics!
            for i in range(nbins):
                histo_new.SetBinContent(i,0)
                histo_new.SetBinError(i,0)
        histo_new.GetXaxis().SetTitle(histos[var][1])
        histo_new.GetYaxis().SetTitle('events')

        #        hist_new.GetXaxis().SetTitle("") #take from histos file
        #hist_new. GetYAxis().SetTitle('events')
        hs.append(histo_new)
        #        if(iname == 'jpsi_x'):
        #    histo_new.SetLineColor(ROOT.kRed+2)
        #    histo_new.SetFillColor(ROOT.kRed+2)
        #else:
        histo_new.SetLineColor(colours[j])
        histo_new.SetFillColor(colours[j])
        maxima.append(histo_new.GetMaximum())
        ths1.Add(histo_new)
        
    #DATA
    his_d = f.Get("shapes_fit_s/" + channel + "/data")
    histo_d = ROOT.TH1F("data","data", nbins, xmin, xmax)
    for i in range(0,his_d.GetN()):
        histo_d.SetBinContent(i+1,his_d.GetPointY(i))
        #    histo_d.Scale(scale)
    data_max = histo_d.GetMaximum()
    histo_d.SetLineColor(ROOT.kBlack)
    ths1.Draw("hist")
    ths1.SetMaximum(1.6*max(sum(maxima), data_max))
    ths1.SetMinimum(0.)
    histo_d.Draw("ep same")

    stats = ths1.GetStack().Last().Clone()
    stats.SetLineColor(0)
    stats.SetFillColor(ROOT.kGray+1)
    stats.SetFillStyle(3344)
    stats.SetMarkerSize(0)
    stats.Draw('E2 SAME')
    
    leg = ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.SetNColumns(3)
    for i,kk in enumerate(sample_names[:-1]):
        leg.AddEntry(hs[i], titles[kk], 'F' if kk!='data' else 'EP')
    leg.AddEntry(histo_d, titles['data'], 'EP')

    leg.AddEntry(stats, 'stat. unc.', 'F')
    leg.Draw('same')
        
    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
    main_pad.cd()
    #rjpsi_value = ROOT.TPaveText(0.7, 0.65, 0.88, 0.72, 'nbNDC')
    #rjpsi_value.AddText('R(J/#Psi) = %.2f' %weights['jpsi_tau'])
    #     rjpsi_value.SetTextFont(62)
    #rjpsi_value.SetFillColor(0)
    #rjpsi_value.Draw('EP')

    ratio_pad.cd()
    ratio = histo_d.Clone()
    ratio.SetName(ratio.GetName()+'_ratio')
    ratio.Divide(stats)
    ratio_stats = stats.Clone()
    ratio_stats.SetName(ratio.GetName()+'_ratiostats')
    ratio_stats.Divide(stats)
    ratio_stats.SetMaximum(1.999) # avoid displaying 2, that overlaps with 0 in the main_pad
    ratio_stats.SetMinimum(0.001) # and this is for symmetry
    ratio_stats.GetYaxis().SetTitle('obs/exp')
    ratio_stats.GetYaxis().SetTitleOffset(0.5)
    ratio_stats.GetYaxis().SetNdivisions(405)
    ratio_stats.GetXaxis().SetLabelSize(3.* ratio.GetXaxis().GetLabelSize())
    ratio_stats.GetYaxis().SetLabelSize(3.* ratio.GetYaxis().GetLabelSize())
    ratio_stats.GetXaxis().SetTitleSize(3.* ratio.GetXaxis().GetTitleSize())
    ratio_stats.GetYaxis().SetTitleSize(3.* ratio.GetYaxis().GetTitleSize())

    norm_stack = ROOT.THStack('norm_stack', '')
    #         import pdb ; pdb.set_trace()
    for i,k in enumerate(hs):
        hh = k.Clone()
        hh.Divide(stats)
        norm_stack.Add(hh)
    #norm_stack.Draw('hist same')
    

    line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(1)
    ratio_stats.Draw('E2')
    norm_stack.Draw('hist same')
    #ratio_stats.Draw('E2 same')
    line.Draw('same')
    ratio.Draw('EP same')
    
    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_postfit/%s/pdf/%s_%s.pdf' %(label, var, channel))
    c1.SaveAs('plots_postfit/%s/png/%s_%s.png' %(label, var, channel))


f.Close()
