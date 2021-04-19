import ROOT
from samples_wf import sample_names
from officialStyle import officialStyle
from cmsstyle import CMS_lumi
import os
from histos import histos

ROOT.gROOT.SetBatch()   
ROOT.gStyle.SetOptStat(0)


histos_folder = '16Apr2021_15h08m54s'
variable = 'Q_sq'
path = '/work/friti/rjpsi_tools/CMSSW_10_6_14/src/RJpsiTools/plotting/plots_ul/'

#output
path_out = path+histos_folder+'/check_sys'
os.system('mkdir -p %s'%path_out)
os.system('mkdir -p %s/%s'%(path_out,variable))

datacard_path = path+histos_folder+'/datacards/datacard_pass_'+variable+'.root'

fin = ROOT.TFile.Open(datacard_path,'r')

hammer_syst = ['bglvar_e0',
               'bglvar_e1',
               'bglvar_e2',
               'bglvar_e3',
               'bglvar_e4',
               'bglvar_e5',
               'bglvar_e6',
               'bglvar_e7',
               'bglvar_e8',
               'bglvar_e9',
               'bglvar_e10',
]

ctau_syst = ['ctau',]

#plotting features
#officialStyle(ROOT.gStyle, ROOT.TGaxis)

c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
c1.SetTicks(True)
c1.SetBottomMargin(0.15)


for sname in sample_names:
    if (sname  !='jpsi_x_mu') and (sname !='jpsi_x') and (sname != 'data'):

        # all have ctau syst
        his_central = fin.Get(sname)
        xmin = his_central.GetBinLowEdge(1)
        xmax = his_central.GetBinLowEdge(his_central.GetNbinsX() + 1)
        nbins = his_central.GetNbinsX()

        histo_central = ROOT.TH1F(sname,sname, nbins, xmin, xmax)

        for i in range(1,his_central.GetNbinsX()+1):
            histo_central.SetBinContent(i,his_central.GetBinContent(i))
            histo_central.SetBinError(i,his_central.GetBinError(i))
        
        for syst in hammer_syst + ctau_syst:
            print(sname,syst)
            maxx = []
            if syst in hammer_syst:
                if sname != 'jpsi_tau' and sname != 'jpsi_mu':
                    continue
                else:
                    syst = sname+'_'+syst
            his_up = fin.Get(sname+'_'+syst+'Up')
            histo_up = ROOT.TH1F(sname+'_'+syst+'Up',sname+'_'+syst+'Up', nbins, xmin, xmax)
            for i in range(1,his_up.GetNbinsX()+1):
                histo_up.SetBinContent(i,his_up.GetBinContent(i))
                histo_up.SetBinError(i,his_up.GetBinError(i))
            maxx.append(histo_up.GetMaximum())
            his_down = fin.Get(sname+'_'+syst+'Down')
            histo_down = ROOT.TH1F(sname+'_'+syst+'Down',sname+'_'+syst+'Down', nbins, xmin, xmax)
            for i in range(1,his_down.GetNbinsX()+1):
                histo_down.SetBinContent(i,his_down.GetBinContent(i))
                histo_down.SetBinError(i,his_down.GetBinError(i))
            maxx.append(histo_down.GetMaximum())
            if syst == 'ctau':
                histo_central.SetTitle(sname +'_'+syst+';'+histos[variable][1]+';events')
            else:
                histo_central.SetTitle(syst+';'+histos[variable][1]+';events')
            #histo_central.GetXaxis().SetTitle(histos[variable][1])
            #histo_central.GetYaxis().SetTitle('events')
            
            histo_central.SetLineColor(ROOT.kBlack)
            histo_central.SetMarkerStyle(8)
            histo_up.SetLineColor(ROOT.kRed)
            histo_up.SetMarkerStyle(8)
            histo_down.SetLineColor(ROOT.kGreen)
            histo_down.SetMarkerStyle(8)
            histo_central.SetMaximum(1.2*max(maxx))
            histo_central.Draw("ep")
            histo_up.Draw("ep same")
            histo_down.Draw("ep same")
            
            leg = ROOT.TLegend(0.1,.75,.45,.90)
            leg.SetBorderSize(0)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)
            leg.AddEntry(histo_central, 'central value',  'EP')
            leg.AddEntry(histo_up, 'up value',  'EP')
            leg.AddEntry(histo_down, 'down value',  'EP')
            leg.Draw('same')
        
            #CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

            c1.Modified()
            c1.Update()
            
            #c1.SaveAs('plots_postfit/%s/pdf/%s_%s.pdf' %(label, var, channel))
            #            c1.SaveAs('plots_postfit/%s/png/%s_%s.png' %(label, var, channel))
            c1.SaveAs(path_out+"/"+variable+"/plot_"+sname+'_'+syst+'.png')
