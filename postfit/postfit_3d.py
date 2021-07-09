from cmsstyle import CMS_lumi
from bokeh.palettes import viridis, all_palettes
import ROOT
from datetime import datetime
import os
from samples import weights, sample_names, titles, colours
from officialStyle import officialStyle
from histos import histos
import math

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)

#### var info ####
vars = ['aa','bb','cc','dd']

#### fit path infos ####
#fit_date = "09Jun2021_09h05m24s" #stats works
#fit_date = "09Jun2021_13h47m12s" #stat does not work
fit_date = "09Jul2021_12h21m26s/"
def mkdirs(label, path_comb):
    '''
    Creates the directories to save the plots
    Copies the input datacards and root files and the fit output
    Creates a simbolic link to the original plot folder
    '''
    os.system('mkdir -p plots_postfit/%s/pdf/'%label)
    os.system('mkdir -p plots_postfit/%s/png/'%label)
    os.system('mkdir -p plots_postfit/%s/inputs/'%label)

    #copy datacard; root files
    os.system("cp "+ path_comb + "fit3d_"+fit_date+ "/datacard.txt " + "plots_postfit/"+label+"/inputs/.")
    for var in vars:
        os.system("cp "+ path_comb +"fit3d_"+fit_date+ "/datacard_pass_"+var+".root plots_postfit/"+label+"/inputs/.")
        os.system("cp "+ path_comb +"fit3d_"+fit_date+ "/datacard_fail_"+var+".root plots_postfit/"+label+"/inputs/.")
    os.system("cp "+ path_comb +"fit3d_"+fit_date+ "/fitDiagnostics.root plots_postfit/"+label+"/inputs/.")

    #symbolic link to original plot folder
    symb_out = os.popen("find "+path_comb + "fit3d_"+fit_date+" . -maxdepth 1 -type l -ls").readlines()[0]
    splitted1 = symb_out.split(" ")
    os.system("ln -s "+ splitted1[-3]+" plots_postfit/"+label+"/inputs/")

if __name__ == "__main__":

    label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')
    path_comb = '/work/friti/combine/CMSSW_10_2_13/src/HiggsAnalysis/CombinedLimit/'
    mkdirs(label,path_comb)

    for nchannel,var in enumerate(vars):
        #take xmin and xmax automatically from histo
        f=ROOT.TFile("plots_postfit/"+label+"/inputs/datacard_pass_"+var+".root","r")
        his = f.Get("jpsi_tau")
        xmin = 0
        nbins = his.GetNbinsX()
        xmax = nbins
        f.Close()
        #plot style
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
        
        for channel in ['ch'+str(nchannel+1),'ch'+str(nchannel+5)]:
            ths1      = ROOT.THStack('stack', '')
            c1.cd()
            main_pad.cd()
            main_pad.SetLogy(False)
            maxima = []
            data_max = 0.
            hs = []
            for j,iname in enumerate(sample_names[:-1] + ['fakes']): #except for data
                histo = f.Get("shapes_fit_s/" + channel + "/" + iname)
                #print(var,histo.GetNbinsX())
                histo_new = ROOT.TH1F(iname+str(j)+channel,"", nbins, xmin, xmax)
                try: 
                    if math.isnan(histo.GetBinError(5)):
                        print(iname + " has a nan uncertainty!")
                except:
                    print("")
                #if histo.GetBinError(i) == np.nan:
                #    print("no")
                for i in range(1,nbins+1):
                    try:
                        histo_new.SetBinContent(i,histo.GetBinContent(i))
                        histo_new.SetBinError(i,histo.GetBinError(i))
                        #histo_new.Scale(scale)
                    except: #if histo is empy from before the fit, combine doesn't save it at all in fitdiagnostics!
                        histo_new.SetBinContent(i,0)
                        histo_new.SetBinError(i,0)
                histo_new.GetXaxis().SetTitle("Unrolled bins")
                histo_new.GetYaxis().SetTitle('events')
                hs.append(histo_new)
                
                histo_new.SetLineColor(colours[iname])
                histo_new.SetFillColor(colours[iname])
                maxima.append(histo_new.GetMaximum())
                ths1.Add(histo_new)
	        
            #DATA
            his_d = f.Get("shapes_fit_s/" + channel + "/data")
            histo_d = ROOT.TH1F("data"+channel,"data"+channel, nbins, xmin, xmax)
            for i in range(0,his_d.GetN()):
                histo_d.SetBinContent(i+1,his_d.GetPointY(i))
                #    histo_d.Scale(scale)
            data_max = histo_d.GetMaximum()
            histo_d.SetLineColor(ROOT.kBlack)
            ths1.Draw("hist")
            ths1.SetMaximum(1.6*max(sum(maxima), data_max))
            ths1.SetMinimum(0.)
            histo_d.Draw("ep same")
            
            #STATS
            stats = ths1.GetStack().Last().Clone()
            stats.SetLineColor(0)
            stats.SetFillColor(ROOT.kGray+1)
            stats.SetFillStyle(3344)
            stats.SetMarkerSize(0)
            stats.Draw('E2 SAME')
            
            #legend
            leg = ROOT.TLegend(0.24,.67,.95,.90)
            leg.SetBorderSize(0)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)
            leg.SetNColumns(3)
            for i,kk in enumerate(sample_names[:-1]+['fakes']):
                leg.AddEntry(hs[i], titles[kk], 'F' if kk!='data' else 'EP')
            leg.AddEntry(histo_d, titles['data'], 'EP')
            leg.AddEntry(stats, 'stat. unc.', 'F')
            leg.Draw('same')
            CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')
            
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
            
            for i,k in enumerate(hs):
                hh = k.Clone()
                hh.Divide(stats)
                norm_stack.Add(hh)
            line = ROOT.TLine(ratio.GetXaxis().GetXmin(), 1., ratio.GetXaxis().GetXmax(), 1.)
            line.SetLineColor(ROOT.kBlack)
            line.SetLineWidth(1)
            ratio_stats.Draw('E2 ')
            norm_stack.Draw('hist same')
            line.Draw('same')
            ratio.Draw('EP same')
            
            c1.Modified()
            c1.Update()
            c1.SaveAs('plots_postfit/%s/pdf/%s_%s.pdf' %(label, var, channel))
            c1.SaveAs('plots_postfit/%s/png/%s_%s.png' %(label, var, channel))
	        
f.Close()
