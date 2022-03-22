import ROOT
from officialStyle import officialStyle
from samples import sample_names, sample_names_explicit_all, sample_names_explicit_jpsimother_compressed, colours, titles
from cmsstyle import CMS_lumi
import os
from histos import histos, histos_hm
import numpy as np

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

def shape_comparison(histos_original, histos_folder, variable, channel, sample_names=sample_names, path = '/work/lmarches/CMS/RJPsi_Tools/CMSSW_10_6_14/src/RJpsiTools/plotting/plots_ul/', verbose = False):

    #print("#######################  "+channel+ "  #####################")
    # output folder
    path_out_1 = path+histos_folder+'/shape_comparison'
    if not os.path.exists(path_out_1) : os.system('mkdir -p %s'%path_out_1)
    #path_out_2 = path_out_1 + '/'+variable
    #if not os.path.exists(path_out_2) : os.system('mkdir -p %s'%path_out_2)
    #print(type(channel),type(path_out_2))
    path_out = path_out_1 + '/'+channel
    if not os.path.exists(path_out) : os.system('mkdir -p %s'%path_out)
    if not os.path.exists(path_out+"/log") : os.system('mkdir -p %s/log'%path_out)

    # Plot 
    c4 = ROOT.TCanvas('c4', '', 700, 700)
    c4.Draw()
    c4.cd()
    c4.SetTicks(True)
    c4.SetBottomMargin(0.15)

    leg = ROOT.TLegend(0.5,.75,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    maxx = []
    i = 0
    histo = {}
    for sample in histos_original:
        #if sample not in samples_comparison:
        #    continue
        his = histos_original[sample]
        xmin = his.GetBinLowEdge(1)
        xmax = his.GetBinLowEdge(his.GetNbinsX() + 1)
        nbins = his.GetNbinsX()
        
        histo[sample] = ROOT.TH1F(sample,sample, nbins, xmin, xmax)
        for i in range(1,his.GetNbinsX()+1):
            histo[sample].SetBinContent(i,his.GetBinContent(i))
            histo[sample].SetBinError(i,his.GetBinError(i))
        
        histo[sample].Scale(1./histo[sample].Integral())
        maxx.append(histo[sample].GetMaximum())
        try:
            histo[sample].SetTitle(';'+histos[variable][1]+';events')
        except:
            histo[sample].SetTitle(';'+histos_hm[variable][1]+';events')
        histo[sample].SetLineColor(colours[sample])
        histo[sample].SetMarkerStyle(8)
        histo[sample].SetMarkerColor(colours[sample])
        histo[sample].SetFillColor(0)
        histo[sample].SetMaximum(1.5*max(maxx))
        c4.cd()

        leg.AddEntry(histo[sample], '%s'%titles[sample],  'ep')
        c4.Modified()
        c4.Update()
    
        i+=1

    for i,sample in enumerate(histo):
        histo[sample].SetMaximum(1.3*max(maxx))
        if i==0:
            histo[sample].Draw("hist")
        else:
            histo[sample].Draw("hist same")

    leg.Draw('same')
    
    CMS_lumi(c4, 4, 0, cmsText = 'CMS', extraText = ' Work in Progress', lumi_13TeV = 'L = 59.7 fb^{-1}', verbose = False)
                
    c4.SaveAs(path_out+"/"+variable+".png")
    c4.SaveAs(path_out+"/"+variable+".pdf")

    c4.SetLogy(True)
    c4.SaveAs(path_out+"/log/"+variable+".png")
    c4.SaveAs(path_out+"/log/"+variable+".pdf")
    #fin.Close()

if __name__ == "__main__":

    histos_folder = '03Mar2022_14h49m37s'
    variable = 'Q_sq'
    shape_comparison(histos_folder, variable,'ch1',sample_names_explicit_jpsimother_compressed )
