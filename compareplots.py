import ROOT
import numpy as np
from bokeh.palettes import viridis, all_palettes
from histos import histos
from cmsstyle import CMS_lumi

from officialStyle import officialStyle
officialStyle(ROOT.gStyle)
ROOT.gStyle.SetTitleOffset(1.5, "Y")
ROOT.gStyle.SetTitleOffset(0.85, "X")
ROOT.gStyle.SetPadLeftMargin(0.20)

ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)   

sample_names = [
    'jpsi_tau_new',
    'jpsi_mu_new' ,
    'jpsi_tau_old',
    'jpsi_mu_old' ,
]

titles = dict()
titles['jpsi_tau_new'] = 'B_{c}#rightarrowJ/#Psi#tau Private'
titles['jpsi_mu_new' ] = 'B_{c}#rightarrowJ/#Psi#mu Private'
titles['jpsi_tau_old'] = 'B_{c}#rightarrowJ/#Psi#tau UL18'
titles['jpsi_mu_old' ] = 'B_{c}#rightarrowJ/#Psi#mu UL18'
   
preselection = ' & '.join([
    'mu1pt>3'               ,
    'mu2pt>3'               ,
    'kpt>2.5'               ,
    'abs(mu1eta)<2.5'       ,
    'abs(mu2eta)<2.5'       ,
    'abs(keta)<2.5'         ,
    'Bsvprob>1e-7'          ,
    'abs(k_dxy)<0.2'        ,
    'abs(mu1_dxy)<0.2'      ,
    'abs(mu2_dxy)<0.2'      ,
    'Bcos2D>0.95'           ,
    'Bmass<6.3'             ,
    'mu1_mediumID>0.5'      ,
    'mu2_mediumID>0.5'      ,
    'k_mediumID>0.5'        ,
    'Bpt_reco>15'           ,
    'abs(mu1_dz-mu2_dz)<0.4', 
    'abs(mu1_dz-k_dz)<0.4'  ,
    'abs(mu2_dz-k_dz)<0.4'  ,
])

preselection_mc = ' & '.join([preselection, 'abs(k_genpdgId)==13'])

samples = dict()
samples['jpsi_tau_new'] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_jpsi_tau_merged.root').Filter(preselection_mc) 
samples['jpsi_mu_new' ] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_jpsi_mu_merged.root' ).Filter(preselection_mc) 
samples['jpsi_tau_old'] = ROOT.RDataFrame('BTommm', '../dataframes_2020Oct19/BcToXToJpsi_is_jpsi_tau_merged.root'       ).Filter(preselection_mc) 
samples['jpsi_mu_old' ] = ROOT.RDataFrame('BTommm', '../dataframes_2020Oct19/BcToXToJpsi_is_jpsi_mu_merged.root'        ).Filter(preselection_mc) 

to_define = [
    ('abs_mu1_dxy' , 'abs(mu1_dxy)'         ),
    ('abs_mu2_dxy' , 'abs(mu2_dxy)'         ),
    ('abs_k_dxy'   , 'abs(k_dxy)'           ),
    ('abs_mu1_dz'  , 'abs(mu1_dz)'          ),
    ('abs_mu2_dz'  , 'abs(mu2_dz)'          ),
    ('abs_k_dz'    , 'abs(k_dz)'            ),
    ('log10_svprob', 'TMath::Log10(Bsvprob)'),
    ('b_iso03_rel' , 'b_iso03/Bpt'          ),
    ('b_iso04_rel' , 'b_iso04/Bpt'          ),
    ('k_iso03_rel' , 'k_iso03/kpt'          ),
    ('k_iso04_rel' , 'k_iso04/kpt'          ),
    ('l1_iso03_rel', 'l1_iso03/mu1pt'       ),
    ('l1_iso04_rel', 'l1_iso04/mu1pt'       ),
    ('l2_iso03_rel', 'l2_iso03/mu2pt'       ),
    ('l2_iso04_rel', 'l2_iso04/mu2pt'       ),
    ('mu1_p4'      , 'ROOT::Math::PtEtaPhiMVector(mu1pt, mu1eta, mu1phi, mu1mass)'),
    ('mu2_p4'      , 'ROOT::Math::PtEtaPhiMVector(mu2pt, mu2eta, mu2phi, mu2mass)'),
    ('mu3_p4'      , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, kmass)'),
    ('jpsi_p4'     , 'mu1_p4+mu2_p4'        ),
    ('jpsi_pt'     , 'jpsi_p4.pt()'         ),
    ('jpsi_eta'    , 'jpsi_p4.eta()'        ),
    ('jpsi_phi'    , 'jpsi_p4.phi()'        ),
    ('jpsi_mass'   , 'jpsi_p4.mass()'       ),
    ('dr12'        , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu2_p4.Vect())'),
    ('dr13'        , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu3_p4.Vect())'),
    ('dr23'        , 'ROOT::Math::VectorUtil::DeltaR(mu2_p4.Vect(), mu3_p4.Vect())'),
    ('dr_jpsi_mu'  , 'ROOT::Math::VectorUtil::DeltaR(jpsi_p4.Vect(), mu3_p4.Vect())'),
    # is there a better way?
    ('maxdr'       , 'dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'),
    ('mindr'       , 'dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'),
]

for k, v in samples.items():
    for new_column, new_definition in to_define:
        samples[k] = samples[k].Define(new_column, new_definition)

# better for categorical data
# colours = list(map(ROOT.TColor.GetColor, all_palettes['Category10'][len(samples)]))
colours = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][len(samples)]))

print ('user defined variables')
print ('='*80)
for i in samples['jpsi_mu_new'].GetDefinedColumnNames(): print(i)
print ('%'*80)

c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()

# CREATE THE SMART POINTERS IN ONE GO AND PRODUCE RESULTS IN ONE SHOT,
# SEE MAX GALLI PRESENTATION
# https://github.com/maxgalli/dask-pyroot-tutorial/blob/master/2_rdf_basics.ipynb
# https://indico.cern.ch/event/882824/contributions/3929999/attachments/2073718/3481850/PyROOT_PyHEP_2020.pdf

# first create all the pointers
print('====> creating pointers to histo')
temp_hists = {}
to_skip = []
for k, v in histos.items():    
    temp_hists[k] = {}
    for kk, vv in samples.items():
        try:
            temp_hists[k]['%s_%s' %(k, kk)] = vv.Histo1D(v[0], k, 'puWeight')
        except:
            print('problem with', k, 'skipping...')
            to_skip.append(k)
            
print('====> now looping')
# then let RDF lazyness work 
for k, v in histos.items():

    if k in to_skip:
        continue
        
    leg = ROOT.TLegend(0.22,.74,.93,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.SetNColumns(2)
    
    for kk, vv in samples.items():
        leg.AddEntry(temp_hists[k]['%s_%s' %(k, kk)].GetValue(), titles[kk], 'L')

    maxima = []
    data_max = 0.
    for i, kv in enumerate(temp_hists[k].items()):
        key = kv[0]
        ihist = kv[1]
        ihist.GetXaxis().SetTitle(v[1])
        ihist.GetYaxis().SetTitle('a.u.')
        ihist.Scale(1./ihist.Integral())
        ihist.SetLineColor(colours[i])
        ihist.SetFillColor(colours[i])
        ihist.SetFillStyle(0) # hollow
        maxima.append(ihist.GetMaximum())
    
    c1.SetLogy(v[2])

    for i, kv in enumerate(temp_hists[k].items()):
        key = kv[0]
        if key=='%s_data'%k: continue
        ihist = kv[1]
        ihist.SetMaximum(2.*max(maxima))
        if not v[2]:
            ihist.SetMinimum(0.)
        ihist.Draw('hist' + 'same'*(i>0))

        if v[2]:
            ihist.SetMaximum(20*max(maxima))
        else:
            ihist.SetMaximum(1.5*max(maxima))

    leg.Draw('same')
            
    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Simulation', lumi_13TeV = '')

    c1.cd()

    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_shapes/pdf/%s.pdf' %k)
    c1.SaveAs('plots_shapes/png/%s.png' %k)
    