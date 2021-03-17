# comparison shapes now from shapes in dataframes of Decemnber (old samples; slightly different production code)
import os
import ROOT
from new_branches import to_define
from selections import preselection, preselection_mc, pass_id, fail_id
from samples import weights, sample_names, titles
from datetime import datetime
from cmsstyle import CMS_lumi


ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)



def make_directories(label):

    os.system('mkdir -p plots_comp/%s/pdf/lin/' %label)
    os.system('mkdir -p plots_comp/%s/png/lin/' %label)
    os.system('mkdir -p plots_comp/%s/fail_region/pdf/lin/' %label)
    os.system('mkdir -p plots_comp/%s/fail_region/png/lin/' %label)

def save_selection(label, preselection,pass_id,fail_id):
    with open('plots_comp/%s/selection.py' %label, 'w') as ff:
        total_expected = 0.
        print("selection = ' & '.join([", file=ff)
        for isel in preselection.split(' & '):
            print("    '%s'," %isel, file=ff)
        
        print('])', file=ff)
        print("pass_id = '%s'"%pass_id,file=ff)
        print("fail_id = '%s'"%fail_id,file=ff)

label = datetime.now().strftime('%d%b%Y_%Hh%Mm%Ss')

make_directories(label)

#NEW
tree_name = 'BTo3Mu'
tree_dir_data = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #data
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2021Mar15' #Bc and Hb



samples_new = dict()
samples_new['jpsi_tau'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_jpsi_tau_merged.root' %(tree_dir))
samples_new['jpsi_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_jpsi_mu_merged.root' %(tree_dir))
samples_new['chic0_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_chic0_mu_merged.root' %(tree_dir))
samples_new['chic1_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_chic1_mu_merged.root' %(tree_dir))
samples_new['chic2_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_chic2_mu_merged.root' %(tree_dir))
samples_new['hc_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_hc_mu_merged.root' %(tree_dir))
samples_new['jpsi_hc'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_jpsi_hc_merged.root' %(tree_dir))
samples_new['psi2s_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_psi2s_mu_merged.root' %(tree_dir))
#samples_new['psi2s_tau'] = ROOT.RDataFrame(tree_name,'%s/BcToJPsiMuMu_is_psi2s_tau_merged.root' %(tree_dir))
samples_new['data'] = ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged.root' %(tree_dir_data))
samples_new['onia'] = ROOT.RDataFrame(tree_name,'%s/HbToJPsiMuMu_ptmax_merged.root' %(tree_dir))

# apply filters on newly defined variables
for k, v in samples_new.items():
    for new_column, new_definition in to_define: 
        if samples_new[k].HasColumn(new_column):
            continue
        samples_new[k] = samples_new[k].Define(new_column, new_definition)

    filter = preselection_mc if k!='data' else preselection
    samples_new[k] = samples_new[k].Filter(filter)


#OLD
tree_name = 'BTommm'
tree_dir = '/pnfs/psi.ch/cms/trivcat/store/user/friti/dataframes_2020Dec09/merged'

samples_old = dict()
samples_old['jpsi_tau'] = ROOT.RDataFrame(tree_name,'%s/BcToJpsiTauNu_ptmax_merged.root' %(tree_dir))
samples_old['jpsi_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToJpsiMuNu_ptmax_merged.root' %(tree_dir))
samples_old['chic0_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_chic0_mu_merged.root' %(tree_dir))
samples_old['chic1_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_chic1_mu_merged.root' %(tree_dir))
samples_old['chic2_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_chic2_mu_merged.root' %(tree_dir))
samples_old['hc_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_hc_mu_merged.root' %(tree_dir))
samples_old['jpsi_hc'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_jpsi_hc_merged.root' %(tree_dir))
samples_old['psi2s_mu'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_psi2s_mu_merged.root' %(tree_dir))
#samples_old['psi2s_tau'] = ROOT.RDataFrame(tree_name,'%s/BcToXToJpsi_is_psi2s_tau_merged.root' %(tree_dir))
samples_old['data'] = ROOT.RDataFrame(tree_name,'%s/data_ptmax_merged.root' %(tree_dir))
samples_old['onia'] = ROOT.RDataFrame(tree_name,'%s/OniaX_ptmax_merged.root' %(tree_dir))


preselection_old = ' & '.join([
    'mu1pt>4',
    'mu2pt>4',
    'kpt>2.5',
    'abs(mu1eta)<2.5',
    'abs(mu2eta)<2.5',
    'abs(keta)<2.5',
    'bvtx_svprob>1e-4',
    'jpsivtx_svprob>1e-2',
    'Bmass<6.3',
    'mu1_mediumID>0.5',
    'mu2_mediumID>0.5',
    'Bpt_reco>15',
    'dr12>0.01',
    'dr13>0.01',
    'dr23>0.01',
    'abs(mu1_dz-mu2_dz)<0.2',
    'abs(mu1_dz-k_dz)<0.2',
    'abs(mu2_dz-k_dz)<0.2',
    'abs(k_dxy)<0.05',
    'abs(mu1_dxy)<0.05',
    'abs(mu2_dxy)<0.05',
    'bvtx_cos2D>0.995',
    #'m_miss_sq>0.5',
    'abs(jpsi_mass-3.0969)<0.1',
    'mmm_p4_par>10',
])

pass_id = 'k_mediumID>0.5'
fail_id = '(!(%s))' % pass_id

preselection_mc_old = ' & '.join([preselection_old, 'abs(k_genpdgId)==13'])

# apply filters on newly defined variables
for k, v in samples_old.items():
    for new_column, new_definition in to_define: 
        if samples_old[k].HasColumn(new_column):
            continue
        if('iso' in new_column):
            continue
        samples_old[k] = samples_old[k].Define(new_column, new_definition)
    
    filter = preselection_mc_old if k!='data' else preselection_old
    samples_old[k] = samples_old[k].Filter(filter)

#############################################################
#### COMPARISON OF THE SHAPES!!!
#############################################################
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()
c1.cd()
main_pad = ROOT.TPad('main_pad', '', 0., 0., 1. , 1.  )
main_pad.Draw()
main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.2)

var = 'Q_sq'
for sname in sample_names:
    c1.cd()
    leg = ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    main_pad.cd()
    main_pad.SetLogy(False)

    #key = k
    temp_hist_old = samples_old[sname].Filter(pass_id).Histo1D(ROOT.RDF.TH1DModel('Q_sq'                 , '', 24,      0,    12), var)
    temp_hist_new = samples_new[sname].Filter(pass_id).Histo1D(ROOT.RDF.TH1DModel('Q_sq'                 , '', 24,      0,    12), var)
    
    temp_hist_old.GetXaxis().SetTitle(var)
    temp_hist_old.GetYaxis().SetTitle('events')
    temp_hist_old.SetLineColor(ROOT.kMagenta)
    #temp_hist_old.SetFillColor(ROOT.kMagenta)

    temp_hist_new.GetXaxis().SetTitle(var)
    temp_hist_new.GetYaxis().SetTitle('events')
    temp_hist_new.SetLineColor(ROOT.kBlue)
    #temp_hist_new.SetFillColor(ROOT.kBlue)

    if(sname != 'data'):
        temp_hist_old.Scale(1./temp_hist_old.Integral())
        temp_hist_new.Scale(1./temp_hist_new.Integral())
    maximum = max(temp_hist_old.GetMaximum(),temp_hist_new.GetMaximum())
    temp_hist_old.SetMaximum(2.*maximum)
    temp_hist_old.Draw('hist')
    temp_hist_new.Draw('hist same')

    leg.AddEntry(temp_hist_old.GetValue(), 'old','F')
    leg.AddEntry(temp_hist_new.GetValue(), 'new','F')

    leg.Draw('same')
    
    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    main_pad.cd()
    
    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_comp/%s/pdf/lin/Q_sq_%s.pdf' %(label, sname))
    c1.SaveAs('plots_comp/%s/png/lin/Q_sq_%s.png' %(label, sname))

    #############FAIL########
    c1.cd()
    leg = ROOT.TLegend(0.24,.67,.95,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    main_pad.cd()
    main_pad.SetLogy(False)
    temp_hist_old = samples_old[sname].Filter(fail_id).Histo1D(ROOT.RDF.TH1DModel('Q_sq'                 , '', 24,      0,    12), var)
    temp_hist_new = samples_new[sname].Filter(fail_id).Histo1D(ROOT.RDF.TH1DModel('Q_sq'                 , '', 24,      0,    12), var)

    temp_hist_old.GetXaxis().SetTitle(var)
    temp_hist_old.GetYaxis().SetTitle('events')
    temp_hist_old.SetLineColor(ROOT.kMagenta)
    #temp_hist_old.SetFillColor(ROOT.kMagenta)

    temp_hist_new.GetXaxis().SetTitle(var)
    temp_hist_new.GetYaxis().SetTitle('events')
    temp_hist_new.SetLineColor(ROOT.kBlue)
    #temp_hist_new.SetFillColor(ROOT.kBlue)

    if(sname != 'data'):
        temp_hist_old.Scale(1./temp_hist_old.Integral())
        temp_hist_new.Scale(1./temp_hist_new.Integral())
    maximum = max(temp_hist_old.GetMaximum(),temp_hist_new.GetMaximum())
    temp_hist_old.SetMaximum(2.*maximum)
    temp_hist_old.Draw('hist')
    temp_hist_new.Draw('hist same')

    leg.AddEntry(temp_hist_old.GetValue(), 'old','F')
    leg.AddEntry(temp_hist_new.GetValue(), 'new','F')

    leg.Draw('same')
    
    CMS_lumi(main_pad, 4, 0, cmsText = 'CMS', extraText = ' Preliminary', lumi_13TeV = '')

    main_pad.cd()
    
    c1.Modified()
    c1.Update()
    
    c1.SaveAs('plots_comp/%s/fail_region/pdf/lin/Q_sq_%s.pdf' %(label, sname))
    c1.SaveAs('plots_comp/%s/fail_region/png/lin/Q_sq_%s.png' %(label, sname))


save_selection(label, preselection,pass_id,fail_id)
