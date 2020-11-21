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
titles['jpsi_tau' ] = 'B_{c}#rightarrow J/#Psi #tau'
titles['jpsi_mu'  ] = 'B_{c}#rightarrow J/#Psi #mu'
titles['jpsi_pi'  ] = 'B_{c}#rightarrow J/#Psi #pi'
titles['psi2s_mu' ] = 'B_{c}#rightarrow #Psi(2S) #mu'
titles['chic0_mu' ] = 'B_{c}#rightarrow #chi_{c0} #mu'
titles['chic1_mu' ] = 'B_{c}#rightarrow #chi_{c1} #mu'
titles['chic2_mu' ] = 'B_{c}#rightarrow #chi_{c2} #mu'
titles['hc_mu'    ] = 'B_{c}#rightarrow h_{c} #mu'
titles['psi2s_tau'] = 'B_{c}#rightarrow #Psi(2S) #tau'
titles['jpsi_3pi' ] = 'B_{c}#rightarrow J/#Psi 3#pi'
titles['jpsi_hc'  ] = 'B_{c}#rightarrow J/#Psi H_{c}'
titles['data'     ] = 'observed'
   
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
for isample_name in sample_names:
    samples[isample_name] = ROOT.RDataFrame('BTommm', 'samples/BcToXToJpsi_is_%s_merged.root' %isample_name).Filter(preselection_mc)

samples['data'] = ROOT.RDataFrame('BTommm', 'samples/data_bc_mmm.root').Filter(preselection)

# @ROOT.Numba.Declare(['float', 'float', 'float'], 'float')
# def maxdr(x, y, z):
#     return max([x, y, z])

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
    ('l1_iso04_rel', 'l1_iso04/mu2pt'       ),
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
    ('maxdr'       , 'dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'),
    ('mindr'       , 'dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'),
]

for k, v in samples.items():
    samples[k] = samples[k].Define('br_weight', '%f' %weights[k])
#     samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight*weightGen')
    samples[k] = samples[k].Define('total_weight', 'br_weight*puWeight' if k!='data' else 'br_weight') # weightGen is suposed to be the lifetime reweigh, but it's broken
    for new_column, new_definition in to_define:
        samples[k] = samples[k].Define(new_column, new_definition)

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
    
    c1.SetLogy(v[2])

    ths1 = ROOT.THStack('stack', '')

    for i, kv in enumerate(temp_hists[k].items()):
        key = kv[0]
        if key=='%s_data'%k: continue
        ihist = kv[1]
        ihist.SetMaximum(2.*max(maxima))
        if not v[2]:
            ihist.SetMinimum(0.)
        ihist.Draw('hist' + 'same'*(i>0))
        ths1.Add(ihist.GetValue())

    ths1.Draw('hist')
    ths1.GetXaxis().SetTitle(v[1])
#     ths1.GetYaxis().SetTitle('a.u.')
    ths1.GetYaxis().SetTitle('events')
    if v[2]:
        ths1.SetMaximum(20*max(sum(maxima), data_max))
    else:
        ths1.SetMaximum(1.5*max(sum(maxima), data_max))

    leg.Draw('same')
    
    temp_hists[k]['%s_data'%k].Draw('EP SAME')
        
    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Simulation', lumi_13TeV = '')

    c1.cd()
    rjpsi_value = ROOT.TPaveText(0.7, 0.68, 0.88, 0.72, 'nbNDC')
    rjpsi_value.AddText('R(J/#Psi) = %.2f' %weights['jpsi_tau'])
#     rjpsi_value.SetTextFont(62)
    rjpsi_value.SetFillColor(0)
    rjpsi_value.Draw()

    c1.Modified()
    c1.Update()
    c1.SaveAs('plots/pdf/%s.pdf' %k)
    c1.SaveAs('plots/png/%s.png' %k)
    