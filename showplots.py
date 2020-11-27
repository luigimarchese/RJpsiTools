import os
import ROOT
import numpy as np
from bokeh.palettes import viridis, all_palettes
from histos import histos
from cmsstyle import CMS_lumi
import pickle

os.system('mkdir -p plots_private/pdf/lin/')
os.system('mkdir -p plots_private/pdf/log/')
os.system('mkdir -p plots_private/png/lin/')
os.system('mkdir -p plots_private/png/log/')

from officialStyle import officialStyle
officialStyle(ROOT.gStyle)
ROOT.gStyle.SetTitleOffset(1.5, "Y")
ROOT.gStyle.SetTitleOffset(0.85, "X")
ROOT.gStyle.SetPadLeftMargin(0.20)

ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)   

sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
#     'jpsi_pi'  ,
    'psi2s_mu' ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'hc_mu'    ,
    'psi2s_tau',
#     'jpsi_3pi' ,
    'jpsi_hc'  ,
    'data'     ,
]

weights = dict()
weights['jpsi_tau' ] = 0.25
weights['jpsi_mu'  ] = 1.
weights['jpsi_pi'  ] = 1.
weights['psi2s_mu' ] = 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['chic0_mu' ] = 0.011600000
weights['chic1_mu' ] = 0.344000000
weights['chic2_mu' ] = 0.195000000
weights['hc_mu'    ] = 0.01
weights['psi2s_tau'] = 0.336000000 + 0.177300000 + 0.032800000 + 0.001300000
weights['jpsi_3pi' ] = 1.
weights['jpsi_hc'  ] = 1.
weights['data'     ] = 1.

# add normalisation factor from Jpsi pi MC
for k, v in weights.items():
    v *= 0.79

titles = dict()
titles['jpsi_tau' ] = 'B_{c}#rightarrowJ/#Psi#tau'
titles['jpsi_mu'  ] = 'B_{c}#rightarrowJ/#Psi#mu'
titles['jpsi_pi'  ] = 'B_{c}#rightarrowJ/#Psi#pi'
titles['psi2s_mu' ] = 'B_{c}#rightarrow#Psi(2S)#mu'
titles['chic0_mu' ] = 'B_{c}#rightarrow#chi_{c0}#mu'
titles['chic1_mu' ] = 'B_{c}#rightarrow#chi_{c1}#mu'
titles['chic2_mu' ] = 'B_{c}#rightarrow#chi_{c2}#mu'
titles['hc_mu'    ] = 'B_{c}#rightarrowh_{c}#mu'
titles['psi2s_tau'] = 'B_{c}#rightarrow#Psi(2S)#tau'
titles['jpsi_3pi' ] = 'B_{c}#rightarrowJ/#Psi3#pi'
titles['jpsi_hc'  ] = 'B_{c}#rightarrowJ/#PsiH_{c}'
titles['data'     ] = 'observed'
   
preselection = ' & '.join([
    'mu1pt>3'               ,
    'mu2pt>3'               ,
    'kpt>2.5'               ,
    'abs(mu1eta)<2.5'       ,
    'abs(mu2eta)<2.5'       ,
    'abs(keta)<2.5'         ,
#     'Bsvprob>1e-7'          ,
    'abs(k_dxy)<0.2'        ,
    'abs(mu1_dxy)<0.2'      ,
    'abs(mu2_dxy)<0.2'      ,
#     'Bcos2D>0.95'           ,
    'Bmass<6.3'             ,
    'mu1_mediumID>0.5'      ,
    'mu2_mediumID>0.5'      ,
    'k_mediumID>0.5'        ,
#     'Bpt_reco>15'           ,
    'abs(mu1_dz-mu2_dz)<0.4', 
    'abs(mu1_dz-k_dz)<0.4'  ,
    'abs(mu2_dz-k_dz)<0.4'  ,
    
#     'bdt_bkg<0.04'          ,
#     'bdt_tau>0.05'          ,
#     'Bsvprob>0.8'           ,
#     'Blxy<0.4'              ,
    'abs(mu1_dz-mu2_dz)<0.2', 
    'abs(mu1_dz-k_dz)<0.2'  ,
    'abs(mu2_dz-k_dz)<0.2'  ,
    'abs(k_dxy)<0.05'       ,
    'abs(mu1_dxy)<0.05'     ,
    'abs(mu2_dxy)<0.05'     ,
    'mu1pt>6'               ,
    'mu2pt>6'               ,
    'kpt>10'                 ,
#     'Bcos2D>0.995'          ,
#     'abs(jpsiK_mass-5.27929)>0.060',
#     'abs(jpsipi_mass-5.27929)>0.060',
#     'abs(Beta)>1.5'         ,
    'jpsi_pt<20',
    'm_miss_sq>0.5',
])

preselection_mc = ' & '.join([preselection, 'abs(k_genpdgId)==13'])

samples = dict()
# for isample_name in sample_names:
#     samples[isample_name] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_%s_merged.root' %isample_name).Filter(preselection_mc)
# 
# samples['data'] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/data_bc_mmm.root').Filter(preselection)

for isample_name in sample_names:
    samples[isample_name] = ROOT.RDataFrame('BTommm', '../samples_20_novembre/samples/BcToXToJpsi_is_%s_enriched.root' %isample_name)

to_define = [
    ('abs_mu1_dxy'  , 'abs(mu1_dxy)'           ),
    ('abs_mu2_dxy'  , 'abs(mu2_dxy)'           ),
    ('abs_k_dxy'    , 'abs(k_dxy)'             ),
    ('abs_mu1_dz'   , 'abs(mu1_dz)'            ),
    ('abs_mu2_dz'   , 'abs(mu2_dz)'            ),
    ('abs_k_dz'     , 'abs(k_dz)'              ),
    ('abs_mu1mu2_dz', 'abs(mu1_dz-mu2_dz)'     ),
    ('abs_mu1k_dz'  , 'abs(mu1_dz-k_dz)'       ),
    ('abs_mu2k_dz'  , 'abs(mu2_dz-k_dz)'       ),
    ('log10_svprob' , 'TMath::Log10(Bsvprob)'  ),
    ('b_iso03_rel'  , 'b_iso03/Bpt'            ),
    ('b_iso04_rel'  , 'b_iso04/Bpt'            ),
    ('k_iso03_rel'  , 'k_iso03/kpt'            ),
    ('k_iso04_rel'  , 'k_iso04/kpt'            ),
    ('l1_iso03_rel' , 'l1_iso03/mu1pt'         ),
    ('l1_iso04_rel' , 'l1_iso04/mu2pt'         ),
    ('l2_iso03_rel' , 'l2_iso03/mu2pt'         ),
    ('l2_iso04_rel' , 'l2_iso04/mu2pt'         ),
    ('mu1_p4'       , 'ROOT::Math::PtEtaPhiMVector(mu1pt, mu1eta, mu1phi, mu1mass)'),
    ('mu2_p4'       , 'ROOT::Math::PtEtaPhiMVector(mu2pt, mu2eta, mu2phi, mu2mass)'),
    ('mu3_p4'       , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, kmass)'),
    ('kaon_p4'      , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, 0.493677)'), # this is at the correct kaon mass
    ('jpsiK_p4'     , 'mu1_p4+mu2_p4+kaon_p4'  ),
    ('jpsiK_mass'   , 'jpsiK_p4.mass()'        ),
    ('jpsiK_pt'     , 'jpsiK_p4.pt()'          ),
    ('jpsiK_eta'    , 'jpsiK_p4.eta()'         ),
    ('jpsiK_phi'    , 'jpsiK_p4.phi()'         ),
    ('pion_p4'      , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, 0.13957018)'), # this is at the correct pion mass
    ('jpsipi_p4'    , 'mu1_p4+mu2_p4+pion_p4'  ),
    ('jpsipi_mass'  , 'jpsipi_p4.mass()'       ),
    ('jpsipi_pt'    , 'jpsipi_p4.pt()'         ),
    ('jpsipi_eta'   , 'jpsipi_p4.eta()'        ),
    ('jpsipi_phi'   , 'jpsipi_p4.phi()'        ),
    ('jpsi_p4'      , 'mu1_p4+mu2_p4'          ),
    ('jpsi_pt'      , 'jpsi_p4.pt()'           ),
    ('jpsi_eta'     , 'jpsi_p4.eta()'          ),
    ('jpsi_phi'     , 'jpsi_p4.phi()'          ),
    ('jpsi_mass'    , 'jpsi_p4.mass()'         ),
    ('dr12'         , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu2_p4.Vect())'),
    ('dr13'         , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu3_p4.Vect())'),
    ('dr23'         , 'ROOT::Math::VectorUtil::DeltaR(mu2_p4.Vect(), mu3_p4.Vect())'),
    ('dr_jpsi_mu'   , 'ROOT::Math::VectorUtil::DeltaR(jpsi_p4.Vect(), mu3_p4.Vect())'),
    # is there a better way?
    ('maxdr'        , 'dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'),
    ('mindr'        , 'dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'),
]

for k, v in samples.items():
    samples[k] = samples[k].Define('br_weight', '%f' %weights[k])
#     samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight*weightGen')
    samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight' if k!='data' else 'br_weight') # weightGen is suposed to be the lifetime reweigh, but it's broken
    for new_column, new_definition in to_define:
        if samples[k].HasColumn(new_column):
            continue
        samples[k] = samples[k].Define(new_column, new_definition)

# apply filters on newly defined variables
for k, v in samples.items():
    filter = preselection_mc if isample_name!='data' else preselection
    samples[k] = samples[k].Filter(filter)

# better for categorical data
# colours = list(map(ROOT.TColor.GetColor, all_palettes['Category10'][len(samples)]))
colours = list(map(ROOT.TColor.GetColor, all_palettes['Spectral'][len(samples)]))

print ('user defined variables')
print ('='*80)
for i in samples['jpsi_mu'].GetDefinedColumnNames(): print(i)
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
for k, v in histos.items():    
    temp_hists[k] = {}
    for kk, vv in samples.items():
        temp_hists[k]['%s_%s' %(k, kk)] = vv.Histo1D(v[0], k, 'total_weight')

print('====> now looping')
# then let RDF lazyness work 
for k, v in histos.items():

    c1.SetLogy(False)

    leg = ROOT.TLegend(0.22,.74,.93,.90)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.SetNColumns(3)
    for kk, vv in samples.items():
        leg.AddEntry(temp_hists[k]['%s_%s' %(k, kk)].GetValue(), titles[kk], 'F' if kk!='data' else 'EP')

    maxima = []
    data_max = 0.
    for i, kv in enumerate(temp_hists[k].items()):
        key = kv[0]
        ihist = kv[1]
        ihist.GetXaxis().SetTitle(v[1])
        ihist.GetYaxis().SetTitle('a.u.')
#         ihist.Scale(1./ihist.Integral())
        ihist.SetLineColor(colours[i] if key!='%s_data'%k else ROOT.kBlack)
        ihist.SetFillColor(colours[i] if key!='%s_data'%k else ROOT.kWhite)
        if key!='%s_data'%k:
            maxima.append(ihist.GetMaximum())
        else:
            data_max = ihist.GetMaximum()
    
    ths1 = ROOT.THStack('stack', '')

    for i, kv in enumerate(temp_hists[k].items()):
        key = kv[0]
        if key=='%s_data'%k: continue
        ihist = kv[1]
        ihist.SetMaximum(2.*max(maxima))
        # ihist.SetMinimum(0.)
        ihist.Draw('hist' + 'same'*(i>0))
        ths1.Add(ihist.GetValue())

    ths1.Draw('hist')
    ths1.GetXaxis().SetTitle(v[1])
#     ths1.GetYaxis().SetTitle('a.u.')
    ths1.GetYaxis().SetTitle('events')
    ths1.SetMaximum(1.5*max(sum(maxima), data_max))
    ths1.SetMinimum(0.)
    
#     import pdb ; pdb.set_trace()
    
    # statistical uncertainty
    stats = ths1.GetStack().Last()
    stats.SetLineColor(0)
    stats.SetFillColor(ROOT.kGray+1)
    stats.SetFillStyle(3344)
    stats.SetMarkerSize(0)
    stats.Draw('E2 SAME')
    leg.AddEntry(stats, 'stat. unc.', 'F')
    
    leg.Draw('same')
    
    temp_hists[k]['%s_data'%k].Draw('EP SAME')
        
    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Preliminary', lumi_13TeV = '')

    c1.cd()
    rjpsi_value = ROOT.TPaveText(0.7, 0.68, 0.88, 0.72, 'nbNDC')
    rjpsi_value.AddText('R(J/#Psi) = %.2f' %weights['jpsi_tau'])
#     rjpsi_value.SetTextFont(62)
    rjpsi_value.SetFillColor(0)
    rjpsi_value.Draw()

    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_private/pdf/lin/%s.pdf' %k)
    c1.SaveAs('plots_private/png/lin/%s.png' %k)
    
    ths1.SetMaximum(20*max(sum(maxima), data_max))
    ths1.SetMinimum(10)
    c1.SetLogy(True)
    c1.Modified()
    c1.Update()
    c1.SaveAs('plots_private/pdf/log/%s.pdf' %k)
    c1.SaveAs('plots_private/png/log/%s.png' %k)

# yields
with open('plots_private/yields.txt', 'w') as ff:
    total_expected = 0.
    for kk, vv in temp_hists[k].items(): 
        if 'data' not in kk:
            total_expected += vv.Integral()
        print(kk.replace(k, '')[1:], '\t\t%.1f' %vv.Integral(), file=ff)
    print('total expected', '\t%.1f' %total_expected, file=ff)

###### Jpsi Mu Nu
###### non UL sample 290620 events in the ntuples
###### UL18 sample 500805 events in the ntuples

###### Jpsi Tau Nu
###### non UL sample 24433 events in the ntuples
###### UL18 sample 38658 events in the ntuples




##############
# The sample here 
# https://cmsweb.cern.ch/das/request?input=/BcToJpsiX_TuneCP5_13TeV-pythia8/manzoni-RunIISummer19UL18_MINIAODSIM_v1-590582b4c566faf0bdd34e5b16afc387/USER&instance=prod/phys03
# contains 20652915 events
# 
# 1./3.15091 = 6554587.404908423
# 
# of which are jpsi_mu
# 
# 6554587.4 jpsi_mu events correspond to 1./0.79 of the 2018 luminosity
# ==>
# 5178124.0 jpsi_mu events correspond to the  2018 luminosity
# 
# tau with RJpsi = 0.25 ==> 5178124.0 *0.25 = 1294531


#  so for this UL18 sample
#  /BcToJPsiMuNu_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen/RunIISummer19UL18MiniAOD-106X_upgrade2018_realistic_v11_L1v1_ext1-v2/MINIAODSIM
# 7475467 events correspond to 1./0.693 the 2018 luminosity
#
#
# this UL18 sample
# https://cmsweb.cern.ch/das/request?input=dataset%3D%2FBcToJPsiTauNu_TuneCP5_13TeV-bcvegpy2-pythia8-evtgen%2FRunIISummer19UL18MiniAOD-106X_upgrade2018_realistic_v11_L1v1_ext1-v2%2FMINIAODSIM&instance=prod/global
# has 6187083 events and therefore 

