import ROOT
import os
from cmsstyle import CMS_lumi
from officialStyle import officialStyle
officialStyle(ROOT.gStyle)
ROOT.gStyle.SetTitleOffset(1.4, "Y")
ROOT.gStyle.SetTitleOffset(0.85, "X")
ROOT.gStyle.SetPadLeftMargin(0.20)

ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)   



# tree_data = ROOT.RDataFrame('tree'   , ['jpsi_pi_splots.root'])
# tree_mc   = ROOT.RDataFrame('tree_mc', ['jpsi_pi_splots.root'])

ff = ROOT.TFile.Open('jpsi_pi_splots.root')
tree = ff.Get('tree')
treemc = ff.Get('treemc')

plot_outdir = 'sPlots'
if not os.path.isdir(plot_outdir):
   os.mkdir(plot_outdir)

# 'mu1_mediumID'         
# 'mu2_mediumID'         
# 'mu1_dxy'              
# 'mu2_dxy'              
# 'k_dxy'                
# 'mu1_dz'               
# 'mu2_dz'               
# 'k_dz'                 

histos = dict()
histos['bvtx_fit_mass'    ] = (ROOT.TH1F('bvtx_fit_mass'    , '', 80,  6    ,   6.6 ), 'J/#Psi#pi mass (GeV)'                             )
histos['Bpt'              ] = (ROOT.TH1F('Bpt'              , '', 15, 15    ,  60   ), 'J/#Psi#pi p_{T} (GeV)'                            )
histos['bvtx_lxy_sig'     ] = (ROOT.TH1F('bvtx_lxy_sig'     , '', 20,  0    , 100   ), 'vtx(#mu_{1}, #mu_{2}, #pi) L_{xy}/#sigma_{L_{xy}}')
histos['bvtx_svprob'      ] = (ROOT.TH1F('bvtx_svprob'      , '', 10,  0    ,   1   ), 'vtx(#mu_{1}, #mu_{2}, #pi) probability'           )
histos['bvtx_cos2D'       ] = (ROOT.TH1F('bvtx_cos2D'       , '', 10,  0.999,   1   ), 'vtx(#mu_{1}, #mu_{2}, #pi) 2D cos#alpha'          )
histos['bvtx_chi2'        ] = (ROOT.TH1F('bvtx_chi2'        , '', 10,  0.   , 100   ), 'vtx(#mu_{1}, #mu_{2}, #pi) #chi^{2}'              )
histos['bvtx_lxy_unc'     ] = (ROOT.TH1F('bvtx_lxy_unc'     , '', 50,  0.   ,   0.03), 'vtx(#mu_{1}, #mu_{2}, #pi) #sigma_{L_{xy}}'       )
histos['jpsivtx_lxy_sig'  ] = (ROOT.TH1F('jpsivtx_lxy_sig'  , '', 20,  0    ,  50   ), 'vtx(#mu_{1}, #mu_{2}) L_{xy}/#sigma_{L_{xy}}'     )
histos['jpsivtx_svprob'   ] = (ROOT.TH1F('jpsivtx_svprob'   , '', 10,  0    ,   1   ), 'vtx(#mu_{1}, #mu_{2}) probability'                )
histos['jpsivtx_cos2D'    ] = (ROOT.TH1F('jpsivtx_cos2D'    , '', 10,  0.999,   1   ), 'vtx(#mu_{1}, #mu_{2}) 2D cos#alpha'               )
histos['jpsivtx_chi2'     ] = (ROOT.TH1F('jpsivtx_chi2'     , '', 40,  0.   ,  10   ), 'vtx(#mu_{1}, #mu_{2}) #chi^{2}'                   )
histos['jpsivtx_lxy_unc'  ] = (ROOT.TH1F('jpsivtx_lxy_unc'  , '', 50,  0.   ,   0.05), 'vtx(#mu_{1}, #mu_{2}) #sigma_{L_{xy}}'            )
histos['mu1pt'            ] = (ROOT.TH1F('mu1pt'            , '', 15,  5    ,  40   ), '#mu_{1} p_{T} (GeV)'                              )
histos['mu2pt'            ] = (ROOT.TH1F('mu2pt'            , '', 15,  5    ,  20   ), '#mu_{2} p_{T} (GeV)'                              )
histos['kpt'              ] = (ROOT.TH1F('kpt'              , '', 15,  2    ,  30   ), '#pi p_{T} (GeV)'                                  )
histos['mu1eta'           ] = (ROOT.TH1F('mu1eta'           , '', 15, -3    ,  3    ), '#mu_{1} #eta'                                     )
histos['mu2eta'           ] = (ROOT.TH1F('mu2eta'           , '', 15, -3    ,  3    ), '#mu_{2} #eta'                                     )
histos['keta'             ] = (ROOT.TH1F('keta'             , '', 15, -3    ,  3    ), '#pi #eta'                                         )
histos['jpsivtx_fit_mass' ] = (ROOT.TH1F('jpsivtx_fit_mass' , '', 20,  2.8  ,  3.4  ), 'J/#Psi mass (GeV)'                                )
histos['jpsivtx_fit_eta'  ] = (ROOT.TH1F('jpsivtx_fit_eta'  , '', 20, -3    ,  3    ), 'J/#Psi #eta'                                      )
c1 = ROOT.TCanvas('c1', '', 700, 700)
c1.Draw()

for k, v in histos.items():


    correlation_histo = ROOT.TH2F('%s_vs_mass' %(k), '', 80, 6., 6.6, v[0].GetNbinsX(), v[0].GetXaxis().GetXmin(), v[0].GetXaxis().GetXmax())
    tree.Draw('%s:bvtx_fit_mass >> %s' %(k, correlation_histo.GetName()), '', '')
    
    correlation_histo.GetYaxis().SetTitle(v[1])
    correlation_histo.GetXaxis().SetTitle('J/#Psi#pi mass (GeV)')
    correlation_histo.SetMarkerStyle(6)

    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Preliminary', lumi_13TeV = '')
    
    c1.Modified()
    c1.Update()
    
    c1.SaveAs('%s/%s_vs_mass_correlation.png' %(plot_outdir, k))
    c1.SaveAs('%s/%s_vs_mass_correlation.pdf' %(plot_outdir, k))


    v[0].GetXaxis().SetTitle(v[1])
    v[0].GetYaxis().SetTitle('events')
    v[0].SetMinimum(0.)
    
    h_all = v[0].Clone()
    h_sig = v[0].Clone()
    h_bkg = v[0].Clone()
    h_mc  = v[0].Clone()

    h_all.SetName( '_'.join([h_all.GetName(), 'all']))
    h_sig.SetName( '_'.join([h_sig.GetName(), 'sig']))
    h_bkg.SetName( '_'.join([h_bkg.GetName(), 'bkg']))
    h_mc .SetName( '_'.join([h_mc .GetName(), 'mc' ]))

    tree.Draw('%s >> %s' %(k, h_all.GetName()), ''       , 'hist')
    tree.Draw('%s >> %s' %(k, h_sig.GetName()), 'nsig_sw', 'hist')
    tree.Draw('%s >> %s' %(k, h_bkg.GetName()), 'nbkg_sw', 'hist')

    h_sig.SetLineColor(ROOT.kRed)
    h_bkg.SetLineColor(ROOT.kBlue)

    h_sig.SetFillColor(ROOT.kRed)
    h_bkg.SetFillColor(ROOT.kBlue)

    h_all.SetMarkerStyle(8)

    ths = ROOT.THStack('ths', '')
    ths.Add(h_sig)
    ths.Add(h_bkg)

    ths.Draw('hist')
    h_all.Draw('EP same')

    # place this after THStack is drawn, else nullptr
    ths.GetXaxis().SetTitle(v[1])
    ths.GetYaxis().SetTitle('events')

    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Preliminary', lumi_13TeV = '')

    leg = ROOT.TLegend(0.6,.7,.88,.88)
    leg.SetHeader('sPlot fits', 'C')
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_sig, 'signal', 'F')
    leg.AddEntry(h_bkg, 'background', 'F')
    leg.AddEntry(h_all, 'observed', 'EP')
    leg.Draw('same')

    c1.Modified()
    c1.Update()
    
    c1.SaveAs('%s/%s_splot.png' %(plot_outdir, k))
    c1.SaveAs('%s/%s_splot.pdf' %(plot_outdir, k))

#     if k=='fit_Bmass':
#         break

    # now compare signal from data (via sPlot) and MC
    treemc.Draw('%s >> %s' %(k, h_mc.GetName()), '', 'hist')

    h_mc.SetLineColor(ROOT.kGreen-2)
    h_mc.SetFillColor(ROOT.kGreen-2)
    h_mc.SetFillStyle(3345)
    h_sig.SetFillStyle(3354)

    h_mc .Scale(1./h_mc .Integral())
    h_sig.Scale(1./h_sig.Integral())
    
    h_mc .Draw('hist')
    h_sig.Draw('hist same')
    
    mymax = 1.2*max(map(ROOT.TH1.GetMaximum, [h_mc, h_sig]))
    
    h_mc .SetMaximum(mymax)
    h_sig.SetMaximum(mymax)
    h_mc .SetMinimum(0.)
    h_sig.SetMinimum(0.)
    h_mc.GetYaxis().SetTitle('a.u.')
    
    CMS_lumi(c1, 4, 0, cmsText = 'CMS', extraText = '   Preliminary', lumi_13TeV = '')
    
    leg = ROOT.TLegend(0.6,.7,.88,.88)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_sig, 'signal from sPlot', 'F')
    leg.AddEntry(h_mc , 'signal MC'        , 'F')
    leg.Draw('same')

    c1.Modified()
    c1.Update()
    
    c1.SaveAs('%s/%s_shapes_splot.png' %(plot_outdir, k))
    c1.SaveAs('%s/%s_shapes_splot.pdf' %(plot_outdir, k))
            
