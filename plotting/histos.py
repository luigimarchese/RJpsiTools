import ROOT
import numpy as np

histos_hm = dict()
histos_hm['jpsivtx_log10_lxy_sig'     ] = (ROOT.RDF.TH1DModel('jpsivtx_log10_lxy_sig'     , '', 20,     -1,     1), 'log_{10} vtx(#mu_{1}, #mu_{2}) L_{xy}/#sigma_{L_{xy}}'         , 1)
histos_hm['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 1,      6.3,    10), '3#mu mass (GeV)'                                               , 0)
histos_hm['norm'                      ] = (ROOT.RDF.TH1DModel('norm'                      , '',  1,      0,     1), 'normalisation'                                                 , 0)

histos = dict()   

#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 15,      5.5,    10), 'q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 20,      0,    4.5), 'q^{2} (GeV^{2})'                                               , 0)
histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 20,      5.5,    10), 'q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq_up'                      ] = (ROOT.RDF.TH1DModel('Q_sq_up'                      , '', 20,      5.5,    10), 'up q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq_down'                      ] = (ROOT.RDF.TH1DModel('Q_sq_down'                      , '', 20,      5.5,    10), 'down q^{2} (GeV^{2})'                                               , 0)
#histos['is_it_bc'                   ] = (ROOT.RDF.TH1DModel('is_it_bc'                   , '',  3,     -1,     2), 'is it bc'                                                      , 0)
#histos['Q_sq_jpsimcorr'                      ] = (ROOT.RDF.TH1DModel('Q_sq_jpsimcorr'                      , '', 20,      5.5,    10), 'q^{2} (GeV^{2})'                                               , 0)
'''
#histos['mez_min'                      ] = (ROOT.RDF.TH1DModel('mez_min'                      , '', 30,      -70,    70), 'min #nu pz [GeV] '                                               , 0)
#histos['mez_max'                      ] = (ROOT.RDF.TH1DModel('mez_max'                      , '', 30,      -70,    70), 'max #nu pz [GeV]'                                               , 0)

#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 24,      0,    10.5), 'q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 1,      0,    10.5), 'q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 50,      -20,    20), 'q^{2} (GeV^{2})'                                               , 0)
#histos['Q_sq'                      ] = (ROOT.RDF.TH1DModel('Q_sq'                      , '', 24,      4.5,    10.5), 'q^{2} (GeV^{2})'                                               , 0)
histos['mu1pt'                     ] = (ROOT.RDF.TH1DModel('mu1pt'                     , '', 50,      0,    40), '#mu_{1} p_{T} (GeV)'                                           , 0)
histos['mu2pt'                     ] = (ROOT.RDF.TH1DModel('mu2pt'                     , '', 50,      0,    20), '#mu_{2} p_{T} (GeV)'                                           , 0)
histos['kpt'                       ] = (ROOT.RDF.TH1DModel('kpt'                       , '', 50,      0,    30), '#mu_{3} p_{T} (GeV)'                                           , 0)

histos['mu1phi'                    ] = (ROOT.RDF.TH1DModel('mu1phi'                    , '', 20, -np.pi, np.pi), '#mu_{1} #phi'                                                  , 0)
histos['mu2phi'                    ] = (ROOT.RDF.TH1DModel('mu2phi'                    , '', 20, -np.pi, np.pi), '#mu_{1} #phi'                                                  , 0)
histos['kphi'                      ] = (ROOT.RDF.TH1DModel('kphi'                      , '', 20, -np.pi, np.pi), '#mu_{1} #phi'                                                  , 0)
histos['mu1eta'                    ] = (ROOT.RDF.TH1DModel('mu1eta'                    , '', 30,     -3,     3), '#mu_{1} #eta'                                                  , 0)
histos['mu2eta'                    ] = (ROOT.RDF.TH1DModel('mu2eta'                    , '', 30,     -3,     3), '#mu_{1} #eta'                                                  , 0)

#histos['keta'                      ] = (ROOT.RDF.TH1DModel('keta'                      , '', 6,     -3,     3), '#mu_{3} #eta'                                                  , 0)

histos['keta'                      ] = (ROOT.RDF.TH1DModel('keta'                      , '', 6,     np.array([-2.5, -1.2, -0.8, 0 , 0.8, 1.2,2.5])), '#mu_{3} #eta'                                                  , 0)

histos['Bpt'                       ] = (ROOT.RDF.TH1DModel('Bpt'                       , '', 50,      10,    50), '3#mu p_{T} (GeV)'                                              , 0)
'''
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 50,      3.2,     5.1), '3#mu mass (GeV)'                                               , 0)
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 50,      5.3,     6.3), '3#mu mass (GeV)'                                               , 0)
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 50,      3.2,     6.3), '3#mu mass (GeV)'                                               , 0)
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 50,      3.2,     10), '3#mu mass (GeV)'                                               , 0)
'''
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 30,      6.3,    10), '3#mu mass (GeV)'                                               , 0)


histos['m13'                     ] = (ROOT.RDF.TH1DModel('m13'                     , '', 100,      2.5,     3.5), 'm13 (GeV)'                                               , 0)
histos['m23'                     ] = (ROOT.RDF.TH1DModel('m23'                     , '', 100,      2.5,     3.5), 'm23 (GeV)'                                               , 0)
#histos['Bmass'                     ] = (ROOT.RDF.TH1DModel('Bmass'                     , '', 60,      3,    10), '3#mu mass (GeV)'                                               , 0)
histos['Beta'                      ] = (ROOT.RDF.TH1DModel('Beta'                      , '', 30,     -3,     3), '3#mu eta'                                                      , 0)
histos['Bphi'                      ] = (ROOT.RDF.TH1DModel('Bphi'                      , '', 20, -np.pi, np.pi), '3#mu #phi'                                                     , 0)
'''
histos['Bpt_reco'                  ] = (ROOT.RDF.TH1DModel('Bpt_reco'                  , '', 50,      0,    80), 'corrected 3#mu p_{T} (GeV)'                                    , 0)
#histos['bc_gen_pt'                  ] = (ROOT.RDF.TH1DModel('bc_gen_pt'                  , '', 50,      0,    80), 'gen 3#mu p_{T} (GeV)'                                    , 0)
#histos['Bpt_reco_down'                  ] = (ROOT.RDF.TH1DModel('Bpt_reco_down'                  , '', 50,      0,    80), 'down 3#mu p_{T} (GeV)'                                    , 0)
#histos['Bpt_reco_up'                  ] = (ROOT.RDF.TH1DModel('Bpt_reco_up'                  , '', 50,      0,    80), 'up 3#mu p_{T} (GeV)'                                    , 0)
'''
histos['abs_mu1_dxy'               ] = (ROOT.RDF.TH1DModel('mu1_dxy'                   , '', 50,      0,   0.06), '#mu_{1} |d_{xy}| (cm)'                                         , 1)
histos['mu1_dxy_sig'               ] = (ROOT.RDF.TH1DModel('mu1_dxy_sig'               , '', 50,      0,    10), '#mu_{1} |d_{xy}|/#sigma_{d_{xy}}'                              , 1)
histos['abs_mu2_dxy'               ] = (ROOT.RDF.TH1DModel('mu2_dxy'                   , '', 50,      0,   0.06), '#mu_{2} |d_{xy}| (cm)'                                         , 1)
histos['mu2_dxy_sig'               ] = (ROOT.RDF.TH1DModel('mu2_dxy_sig'               , '', 50,      0,    10), '#mu_{2} |d_{xy}|/#sigma_{d_{xy}}'                              , 1)
histos['abs_k_dxy'                 ] = (ROOT.RDF.TH1DModel('k_dxy'                     , '', 50,      0,   0.06), '#mu_{3} |d_{xy}| (cm)'                                         , 1)
histos['k_dxy_sig'                 ] = (ROOT.RDF.TH1DModel('k_dxy_sig'                 , '', 50,      0,    10), '#mu_{3} |d_{xy}|/#sigma_{d_{xy}}'                              , 1)
histos['abs_mu1_dz'                ] = (ROOT.RDF.TH1DModel('mu1_dz'                    , '', 50,      0,   0.4), '#mu_{1} |d_{z}| (cm)'                                          , 1)
histos['mu1_dz_sig'                ] = (ROOT.RDF.TH1DModel('mu1_dz_sig'                , '', 50,      0,    10), '#mu_{1} |d_{z}|/#sigma_{d_{z}}'                                , 1)
histos['abs_mu2_dz'                ] = (ROOT.RDF.TH1DModel('mu2_dz'                    , '', 50,      0,   0.4), '#mu_{2} |d_{z}| (cm)'                                          , 1)
histos['mu2_dz_sig'                ] = (ROOT.RDF.TH1DModel('mu2_dz_sig'                , '', 50,      0,    10), '#mu_{2} |d_{z}|/#sigma_{d_{z}}'                                , 1)
histos['abs_k_dz'                  ] = (ROOT.RDF.TH1DModel('k_dz'                      , '', 50,      0,   0.4), '#mu_{3} |d_{z}| (cm)'                                          , 1)
histos['k_dz_sig'                  ] = (ROOT.RDF.TH1DModel('k_dz_sig'                  , '', 50,      0,    10), '#mu_{3} |d_{z}|/#sigma_{d_{z}}'                                , 1)
histos['k_dxyErr'                  ] = (ROOT.RDF.TH1DModel('k_dxyErr'                  , '', 50,      0,    0.01), '#mu_{3} #sigma_{d_{xy}}'                                , 1)
histos['mu1_dxyErr'                  ] = (ROOT.RDF.TH1DModel('mu1_dxyErr'                  , '', 50,      0,    0.01), '#mu_{1} #sigma_{d_{xy}}'                                , 1)
histos['mu2_dxyErr'                  ] = (ROOT.RDF.TH1DModel('mu2_dxyErr'                  , '', 50,      0,    0.01), '#mu_{2} #sigma_{d_{xy}}'                                , 1)
'''
'''
# histos['Blxy_sig'                  ] = (ROOT.RDF.TH1DModel('Blxy_sig'                  , '', 50,      0,    60), 'L_{xy}/#sigma_{L_{xy}}'                                        , 1)
# histos['Blxy'                      ] = (ROOT.RDF.TH1DModel('Blxy'                      , '',100,      0,     2), 'L_{xy} (cm)'                                                   , 1)
# histos['Blxy_unc'                  ] = (ROOT.RDF.TH1DModel('Blxy_unc'                  , '',100,      0,  0.02), '#sigma_{L_{xy}} (cm)'                                          , 1)
# histos['Bsvprob'                   ] = (ROOT.RDF.TH1DModel('Bsvprob'                   , '', 50,      0,     1), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) probability'                    , 0)
histos['bvtx_chi2'                 ] = (ROOT.RDF.TH1DModel('bvtx_chi2'                 , '', 50,      0,    50), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) #chi^{2}'                       , 1)
histos['bvtx_lxy'                  ] = (ROOT.RDF.TH1DModel('bvtx_lxy'                  , '',100,      0,     2), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) L_{xy} (cm)'                                                   , 1)
#histos['bvtx_lxy_sig'              ] = (ROOT.RDF.TH1DModel('bvtx_lxy_sig'              , '', 50,      0,    60), 'L_{xy}/#sigma_{L_{xy}}'                                        , 1)
histos['bvtx_lxy_sig'              ] = (ROOT.RDF.TH1DModel('bvtx_lxy_sig'              , '', 50,      0,    30), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) L_{xy}/#sigma_{L_{xy}}'                                        , 1)
histos['bvtx_lxy_unc'              ] = (ROOT.RDF.TH1DModel('bvtx_lxy_unc'              , '', 60,      0,  0.02), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) #sigma_{L_{xy}} (cm)'                                          , 1)
histos['bvtx_lxy_sig_corr'         ] = (ROOT.RDF.TH1DModel('bvtx_lxy_sig_corr'         , '', 50,      0,    60), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) corrected L_{xy}/#sigma_{L_{xy}}'                              , 1)
histos['bvtx_lxy_unc_corr'         ] = (ROOT.RDF.TH1DModel('bvtx_lxy_unc_corr'         , '', 60,      0,  0.02), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) corrected #sigma_{L_{xy}} (cm)'                                , 1)
histos['bvtx_log10_lxy_sig_corr'   ] = (ROOT.RDF.TH1DModel('bvtx_log10_lxy_sig_corr'   , '', 51,     -2,     2), 'corrected log_{10} vtx(#mu_{1}, #mu_{2}, #mu_{3}) L_{xy}/#sigma_{L_{xy}}', 1)

histos['bvtx_svprob'               ] = (ROOT.RDF.TH1DModel('bvtx_svprob'               , '', 50,      0,     1), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) probability'                    , 0)
histos['bvtx_log10_svprob'         ] = (ROOT.RDF.TH1DModel('bvtx_log10_svprob'         , '', 51,     -8,     1), 'log_{10} vtx(#mu_{1}, #mu_{2}, #mu_{3}) probability'           , 1)
histos['bvtx_log10_lxy'            ] = (ROOT.RDF.TH1DModel('bvtx_log10_lxy'            , '', 51,     -4,     1), 'log_{10} vtx(#mu_{1}, #mu_{2}, #mu_{3}) L_{xy}'                , 1)
histos['bvtx_log10_lxy_sig'        ] = (ROOT.RDF.TH1DModel('bvtx_log10_lxy_sig'        , '', 51,     -2,     2), 'log_{10} vtx(#mu_{1}, #mu_{2}, #mu_{3}) L_{xy}/#sigma_{L_{xy}}', 1)
#histos['bvtx_cos2D'                ] = (ROOT.RDF.TH1DModel('bvtx_cos2D'                , '',100,    0.9,     1), '2D cos#alpha'                                                  , 1)
histos['bvtx_cos2D'                ] = (ROOT.RDF.TH1DModel('bvtx_cos2D'                , '',100,    0.995,     1), 'mu_{1}, #mu_{2}, #mu_{3}) 2D cos#alpha'                                                  , 1)
histos['jpsivtx_chi2'              ] = (ROOT.RDF.TH1DModel('jpsivtx_chi2'              , '', 50,      0,    50), 'vtx(#mu_{1}, #mu_{2}) #chi^{2}'                                , 1)
#histos['jpsivtx_lxy_sig'           ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy_sig'           , '', 50,      0,    60), 'L_{xy}/#sigma_{L_{xy}}'                                        , 1)
histos['jpsivtx_lxy_sig'           ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy_sig'           , '', 50,      0,    30), 'vtx(#mu_{1}, #mu_{2}) L_{xy}/#sigma_{L_{xy}}'                                        , 1)
histos['jpsivtx_lxy'               ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy'               , '',100,      0,     2), 'vtx(#mu_{1}, #mu_{2}) L_{xy} (cm)'                                                   , 1)
histos['jpsivtx_lxy_unc'           ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy_unc'           , '', 60,      0,  0.02), 'vtx(#mu_{1}, #mu_{2}) #sigma_{L_{xy}} (cm)'                                          , 1)
histos['jpsivtx_lxy_unc_corr'      ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy_unc_corr'      , '', 60,      0,  0.02), 'vtx(#mu_{1}, #mu_{2}) corrected #sigma_{L_{xy}} (cm)'                                          , 1)
histos['jpsivtx_lxy_sig_corr'      ] = (ROOT.RDF.TH1DModel('jpsivtx_lxy_sig_corr'      , '', 50,      0,    60), 'vtx(#mu_{1}, #mu_{2}) corrected L_{xy}/#sigma_{L_{xy}}'                                        , 1)

histos['jpsivtx_log10_lxy_sig_corr'] = (ROOT.RDF.TH1DModel('jpsivtx_log10_lxy_sig_corr', '', 51,     -2,     2), 'corrected log_{10} vtx(#mu_{1}, #mu_{2}) L_{xy}/#sigma_{L_{xy}}'         , 1)

histos['jpsivtx_svprob'            ] = (ROOT.RDF.TH1DModel('jpsivtx_svprob'            , '', 50,      0,     1), 'vtx(#mu_{1}, #mu_{2}) probability'                             , 0)
histos['jpsivtx_log10_svprob'      ] = (ROOT.RDF.TH1DModel('jpsivtx_log10_svprob'      , '', 51,     -8,     1), 'log_{10} vtx(#mu_{1}, #mu_{2}) probability'                    , 1)

histos['jpsivtx_log10_lxy'         ] = (ROOT.RDF.TH1DModel('jpsivtx_log10_lxy'         , '', 51,     -4,     1), 'log_{10} vtx(#mu_{1}, #mu_{2}) L_{xy}'                         , 1)

histos['jpsivtx_log10_lxy_sig'     ] = (ROOT.RDF.TH1DModel('jpsivtx_log10_lxy_sig'     , '', 51,     -2,     2), 'log_{10} vtx(#mu_{1}, #mu_{2}) L_{xy}/#sigma_{L_{xy}}'         , 1)

#histos['jpsivtx_cos2D'             ] = (ROOT.RDF.TH1DModel('jpsivtx_cos2D'             , '',100,    0.9,     1), '2D cos#alpha'                                                  , 1)
histos['jpsivtx_cos2D'             ] = (ROOT.RDF.TH1DModel('jpsivtx_cos2D'             , '',100,    0.99,     1), 'vtx(#mu_{1}, #mu_{2}) 2D cos#alpha'                                                  , 1)

#histos['m_miss_sq'                 ] = (ROOT.RDF.TH1DModel('m_miss_sq'                 , '', 50,      0,    9), 'm^{2}_{miss} (GeV^{2})'                                        , 0)
histos['m_miss_sq'                 ] = (ROOT.RDF.TH1DModel('m_miss_sq'                 , '', 50,      1.7,    9), 'm^{2}_{miss} (GeV^{2})'                                        , 0)



histos['m2missjpsik'               ] = (ROOT.RDF.TH1DModel('m2missjpsik'               , '', 50,      0,    12), 'm^{2}_{miss} (GeV^{2})'                                        , 0)
histos['m2missjpsipi'              ] = (ROOT.RDF.TH1DModel('m2missjpsipi'              , '', 50,      0,    12), 'm^{2}_{miss} (GeV^{2})'                                        , 0)
histos['q2jpsik'                   ] = (ROOT.RDF.TH1DModel('q2jpsik'                   , '', 50,      0,    12), 'q^{2} (GeV^{2})'                                               , 0)
histos['q2jpsipi'                  ] = (ROOT.RDF.TH1DModel('q2jpsipi'                  , '', 50,      0,    12), 'q^{2} (GeV^{2})'                                               , 0)
histos['pt_var'                    ] = (ROOT.RDF.TH1DModel('pt_var'                    , '', 50,      0,    50), 'p_{T}^{var} (GeV)'                                             , 0)

#histos['pt_miss_vec'               ] = (ROOT.RDF.TH1DModel('pt_miss_vec'               , '', 50,      0,    50), 'vector p_{T}^{miss} (GeV)'                                     , 0)
histos['pt_miss_vec'               ] = (ROOT.RDF.TH1DModel('pt_miss_vec'               , '', 50,      0,    30), 'vector p_{T}^{miss} (GeV)'                                     , 0)
histos['pt_miss_scal'              ] = (ROOT.RDF.TH1DModel('pt_miss_scal'              , '', 60,    -10,    50), 'scalar p_{T}^{miss} (GeV)'                                     , 0)


#histos['E_mu_star'                 ] = (ROOT.RDF.TH1DModel('E_mu_star'                 , '', 50,      0.3,   2.3), 'E_{#mu_{3}}* (GeV)'                                            , 0)
histos['E_mu_star'                 ] = (ROOT.RDF.TH1DModel('E_mu_star'                 , '', 50,      0.3,   1.65), 'E_{#mu_{3}}* (GeV)'                                            , 0)

histos['E_mu_canc'                 ] = (ROOT.RDF.TH1DModel('E_mu_canc'                 , '', 50,      0,     2.6), 'E_{#mu_{3}}^{#} (GeV)'                                         , 0)
#histos['E_mu_canc'                 ] = (ROOT.RDF.TH1DModel('E_mu_canc'                 , '', 50,      0,     6), 'E_{#mu_{3}}^{#} (GeV)'                                         , 0)
#histos['E_mu_star'                 ] = (ROOT.RDF.TH1DModel('E_mu_star'                 , '', 50,      0,   10), 'E_{#mu_{3}}* (GeV)'                                            , 0)
#histos['E_mu_canc'                 ] = (ROOT.RDF.TH1DModel('E_mu_canc'                 , '', 50,      0,     10), 'E_{#mu_{3}}^{#} (GeV)'                                         , 0)

histos['mu1_mediumID'              ] = (ROOT.RDF.TH1DModel('mu1_mediumID'              , '',  2,      0,     2), '#mu_{1} mediumID'                                              , 0)
histos['mu2_mediumID'              ] = (ROOT.RDF.TH1DModel('mu2_mediumID'              , '',  2,      0,     2), '#mu_{2} mediumID'                                              , 0)
histos['k_mediumID'                ] = (ROOT.RDF.TH1DModel('k_mediumID'                , '',  2,      0,     2), '#mu_{3} mediumID'                                              , 0)
histos['mu1_tightID'               ] = (ROOT.RDF.TH1DModel('mu1_tightID'               , '',  2,      0,     2), '#mu_{1} tightID'                                               , 0)
histos['mu2_tightID'               ] = (ROOT.RDF.TH1DModel('mu2_tightID'               , '',  2,      0,     2), '#mu_{2} tightID'                                               , 0)
histos['k_tightID'                 ] = (ROOT.RDF.TH1DModel('k_tightID'                 , '',  2,      0,     2), '#mu_{3} tightID'                                               , 0)
histos['mu1_softID'                ] = (ROOT.RDF.TH1DModel('mu1_softID'                , '',  2,      0,     2), '#mu_{1} softID'                                                , 0)
histos['mu2_softID'                ] = (ROOT.RDF.TH1DModel('mu2_softID'                , '',  2,      0,     2), '#mu_{2} softID'                                                , 0)
histos['k_softID'                  ] = (ROOT.RDF.TH1DModel('k_softID'                  , '',  2,      0,     2), '#mu_{3} softID'                                                , 0)
histos['b_iso03'                   ] = (ROOT.RDF.TH1DModel('b_iso03'                   , '', 50,      0,    20), '3-#mu bpark I^{abs}_{R=0.3}'                                   , 0)
histos['b_iso04'                   ] = (ROOT.RDF.TH1DModel('b_iso04'                   , '', 50,      0,    20), '3-#mu bpark I^{abs}_{R=0.4}'                                   , 0)
histos['k_iso03'                   ] = (ROOT.RDF.TH1DModel('k_iso03'                   , '', 50,      0,    20), '#mu_{3} bpark I^{abs}_{R=0.3}'                                 , 0)
histos['k_iso04'                   ] = (ROOT.RDF.TH1DModel('k_iso04'                   , '', 50,      0,    20), '#mu_{3} bpark I^{abs}_{R=0.4}'                                 , 0)
histos['mu1_iso03'                 ] = (ROOT.RDF.TH1DModel('mu1_iso03'                 , '', 50,      0,    20), '#mu_{1} bpark I^{abs}_{R=0.3}'                                 , 0)
histos['mu1_iso04'                 ] = (ROOT.RDF.TH1DModel('mu1_iso04'                 , '', 50,      0,    20), '#mu_{1} bpark I^{abs}_{R=0.4}'                                 , 0)
histos['mu2_iso03'                 ] = (ROOT.RDF.TH1DModel('mu2_iso03'                 , '', 50,      0,    20), '#mu_{2} bpark I^{abs}_{R=0.3}'                                 , 0)
histos['mu2_iso04'                 ] = (ROOT.RDF.TH1DModel('mu2_iso04'                 , '', 50,      0,    20), '#mu_{2} bpark I^{abs}_{R=0.4}'                                 , 0)
histos['k_raw_db_corr_iso03'       ] = (ROOT.RDF.TH1DModel('k_raw_db_corr_iso03'       , '', 50,      0,    20), '#mu_{3} #Delta#beta-corr. I^{abs}_{R=0.3}'                     , 0)
histos['k_raw_db_corr_iso04'       ] = (ROOT.RDF.TH1DModel('k_raw_db_corr_iso04'       , '', 50,      0,    30), '#mu_{3} #Delta#beta-corr. I^{abs}_{R=0.4}'                     , 0)
histos['k_raw_ch_pfiso03'          ] = (ROOT.RDF.TH1DModel('k_raw_ch_pfiso03'          , '', 50,      0,    20), '#mu_{3} PF charged I^{abs}_{R=0.3}'                            , 0)
histos['k_raw_ch_pfiso04'          ] = (ROOT.RDF.TH1DModel('k_raw_ch_pfiso04'          , '', 50,      0,    20), '#mu_{3} PF charged I^{abs}_{R=0.4}'                            , 0)
histos['k_raw_n_pfiso03'           ] = (ROOT.RDF.TH1DModel('k_raw_n_pfiso03'           , '', 50,      0,    20), '#mu_{3} PF neutral I^{abs}_{R=0.3}'                            , 0)
histos['k_raw_n_pfiso04'           ] = (ROOT.RDF.TH1DModel('k_raw_n_pfiso04'           , '', 50,      0,    20), '#mu_{3} PF neutral I^{abs}_{R=0.4}'                            , 0)
histos['k_raw_pho_pfiso03'         ] = (ROOT.RDF.TH1DModel('k_raw_pho_pfiso03'         , '', 50,      0,    20), '#mu_{3} PF #gamma I^{abs}_{R=0.3}'                             , 0)
histos['k_raw_pho_pfiso04'         ] = (ROOT.RDF.TH1DModel('k_raw_pho_pfiso04'         , '', 50,      0,    20), '#mu_{3} PF #gamma I^{abs}_{R=0.4}'                             , 0)
histos['k_raw_pu_pfiso03'          ] = (ROOT.RDF.TH1DModel('k_raw_pu_pfiso03'          , '', 50,      0,    20), '#mu_{3} PF PU I^{abs}_{R=0.3}'                                 , 0)
histos['k_raw_pu_pfiso04'          ] = (ROOT.RDF.TH1DModel('k_raw_pu_pfiso04'          , '', 50,      0,    20), '#mu_{3} PF PU I^{abs}_{R=0.4}'                                 , 0)
histos['k_raw_trk_iso03'           ] = (ROOT.RDF.TH1DModel('k_raw_trk_iso03'           , '', 50,      0,    20), '#mu_{3} track I^{abs}_{R=0.3}'                                 , 0)
histos['k_raw_trk_iso05'           ] = (ROOT.RDF.TH1DModel('k_raw_trk_iso05'           , '', 50,      0,    20), '#mu_{3} track I^{abs}_{R=0.5}'                                 , 0)
# histos['raw_rho_corr_iso03'        ] = (ROOT.RDF.TH1DModel('raw_rho_corr_iso03'        , '', 50,      0,    20), '#mu_{1} I^{abs}_{R=0.3}'                                                 , 0)
# histos['raw_rho_corr_iso04'        ] = (ROOT.RDF.TH1DModel('raw_rho_corr_iso04'        , '', 50,      0,    20), '#mu_{1} I^{abs}_{R=0.4}'                                                 , 0)
histos['b_iso03_rel'               ] = (ROOT.RDF.TH1DModel('b_iso03_rel'               , '', 50,      0,     2), '3-#mu bpark I^{rel}_{R=0.3}'                                   , 0)
histos['b_iso04_rel'               ] = (ROOT.RDF.TH1DModel('b_iso04_rel'               , '', 50,      0,     2), '3-#mu bpark I^{rel}_{R=0.4}'                                   , 0)
histos['k_iso03_rel'               ] = (ROOT.RDF.TH1DModel('k_iso03_rel'               , '', 50,      0,     2), '#mu_{3} bpark I^{rel}_{R=0.3}'                                 , 0)
histos['k_iso04_rel'               ] = (ROOT.RDF.TH1DModel('k_iso04_rel'               , '', 50,      0,     2), '#mu_{3} bpark I^{rel}_{R=0.4}'                                 , 0)
histos['mu1_iso03_rel'             ] = (ROOT.RDF.TH1DModel('mu1_iso03_rel'             , '', 50,      0,     2), '#mu_{1} bpark I^{rel}_{R=0.3}'                                 , 0)
histos['mu1_iso04_rel'             ] = (ROOT.RDF.TH1DModel('mu1_iso04_rel'             , '', 50,      0,     2), '#mu_{1} bpark I^{rel}_{R=0.4}'                                 , 0)
histos['mu2_iso03_rel'             ] = (ROOT.RDF.TH1DModel('mu2_iso03_rel'             , '', 50,      0,     2), '#mu_{2} bpark I^{rel}_{R=0.3}'                                 , 0)
histos['mu2_iso04_rel'             ] = (ROOT.RDF.TH1DModel('mu2_iso04_rel'             , '', 50,      0,     2), '#mu_{2} bpark I^{rel}_{R=0.4}'                                 , 0)

histos['k_raw_db_corr_iso03_rel'   ] = (ROOT.RDF.TH1DModel('k_raw_db_corr_iso03_rel'   , '', 50,      0,     100), '#mu_{3} #Delta#beta-corr. I^{rel}_{R=0.3}'                     , 0)
#histos['k_raw_db_corr_iso03_rel'   ] = (ROOT.RDF.TH1DModel('k_raw_db_corr_iso03_rel'   , '', 1,      0,     100), '#mu_{3} #Delta#beta-corr. I^{rel}_{R=0.3}'                     , 0)

histos['k_raw_db_corr_iso04_rel'   ] = (ROOT.RDF.TH1DModel('k_raw_db_corr_iso04_rel'   , '', 50,      0,     2), '#mu_{3} #Delta#beta-corr. I^{rel}_{R=0.4}'                     , 0)
histos['k_raw_ch_pfiso03_rel'      ] = (ROOT.RDF.TH1DModel('k_raw_ch_pfiso03_rel'      , '', 50,      0,     2), '#mu_{3} PF charged I^{rel}_{R=0.3}'                            , 0)
histos['k_raw_ch_pfiso04_rel'      ] = (ROOT.RDF.TH1DModel('k_raw_ch_pfiso04_rel'      , '', 50,      0,     2), '#mu_{3} PF charged I^{rel}_{R=0.4}'                            , 0)
histos['k_raw_n_pfiso03_rel'       ] = (ROOT.RDF.TH1DModel('k_raw_n_pfiso03_rel'       , '', 50,      0,     2), '#mu_{3} PF neutral I^{rel}_{R=0.3}'                            , 0)
histos['k_raw_n_pfiso04_rel'       ] = (ROOT.RDF.TH1DModel('k_raw_n_pfiso04_rel'       , '', 50,      0,     2), '#mu_{3} PF neutral I^{rel}_{R=0.4}'                            , 0)
histos['k_raw_pho_pfiso03_rel'     ] = (ROOT.RDF.TH1DModel('k_raw_pho_pfiso03_rel'     , '', 50,      0,     2), '#mu_{3} PF #gamma I^{rel}_{R=0.3}'                             , 0)
histos['k_raw_pho_pfiso04_rel'     ] = (ROOT.RDF.TH1DModel('k_raw_pho_pfiso04_rel'     , '', 50,      0,     2), '#mu_{3} PF #gamma I^{rel}_{R=0.4}'                             , 0)
histos['k_raw_pu_pfiso03_rel'      ] = (ROOT.RDF.TH1DModel('k_raw_pu_pfiso03_rel'      , '', 50,      0,     2), '#mu_{3} PF PU I^{rel}_{R=0.3}'                                 , 0)
histos['k_raw_pu_pfiso04_rel'      ] = (ROOT.RDF.TH1DModel('k_raw_pu_pfiso04_rel'      , '', 50,      0,     2), '#mu_{3} PF PU I^{rel}_{R=0.4}'                                 , 0)
histos['k_raw_trk_iso03_rel'       ] = (ROOT.RDF.TH1DModel('k_raw_trk_iso03_rel'       , '', 50,      0,     2), '#mu_{3} track I^{rel}_{R=0.3}'                                 , 0)
histos['k_raw_trk_iso05_rel'       ] = (ROOT.RDF.TH1DModel('k_raw_trk_iso05_rel'       , '', 50,      0,     2), '#mu_{3} track I^{rel}_{R=0.5}'                                 , 0)
# histos['raw_rho_corr_iso03_rel   ' ] = (ROOT.RDF.TH1DModel('raw_rho_corr_iso03_rel'    , '', 50,      0,     2), '#mu_{1} I^{rel}_{R=0.3}'                                       , 0)
# histos['raw_rho_corr_iso04_rel   ' ] = (ROOT.RDF.TH1DModel('raw_rho_corr_iso04_rel'    , '', 50,      0,     2), '#mu_{1} I^{rel}_{R=0.4}'                                       , 0)
histos['Bcharge'                   ] = (ROOT.RDF.TH1DModel('Bcharge'                   , '',  3,     -1,     2), 'B charge'                                                      , 0)
# histos['mll_raw'                   ] = (ROOT.RDF.TH1DModel('mll_raw'                   , '', 50,    2.5,   3.5), 'm(#mu_{1}, #mu_{2}) (GeV)'                                     , 0)
histos['DR_mu1mu2'                 ] = (ROOT.RDF.TH1DModel('DR_mu1mu2'                 , '', 50,      0,   1.2), '#DeltaR(#mu_{1}, #mu_{2})'                                     , 0)
# histos['jpsi_chi2'                 ] = (ROOT.RDF.TH1DModel('jpsi_chi2'                 , '', 80,      0,  20. ), 'vtx(#mu_{1}, #mu_{2}) #chi^{2}'                                , 0)
# histos['Bchi2'                     ] = (ROOT.RDF.TH1DModel('b_chi2'                    , '', 80,      0,  35. ), 'vtx(#mu_{1}, #mu_{2}, #mu_{3}) #chi^{2}'                       , 0)

#histos['jpsi_mass'                 ] = (ROOT.RDF.TH1DModel('jpsi_mass'                 , '', 50,    2.9,   3.2), 'm(#mu_{1}, #mu_{2}) (GeV)'                                     , 0)

#histos['jpsivtx_fit_mass'                 ] = (ROOT.RDF.TH1DModel('jpsivtx_fit_mass'                 , '', 50,    2.9,   3.2), 'm(#mu_{1}, #mu_{2}) (GeV)'                                     , 0)
histos['jpsivtx_fit_mass'                 ] = (ROOT.RDF.TH1DModel('jpsivtx_fit_mass'                 , '', 50,    3,   3.2), 'm(#mu_{1}, #mu_{2}) (GeV)'                                     , 0)
histos['jpsivtx_fit_mass_corr'                 ] = (ROOT.RDF.TH1DModel('jpsivtx_fit_mass_corr'                 , '', 50,    2.9,   3.2), 'corrected m(#mu_{1}, #mu_{2}) (GeV)'                                     , 0)

histos['jpsi_pt'                   ] = (ROOT.RDF.TH1DModel('jpsi_pt'                   , '', 50,      0,    50), '(#mu_{1}, #mu_{2}) p_{T} (GeV)'                                , 0)
histos['jpsi_eta'                  ] = (ROOT.RDF.TH1DModel('jpsi_eta'                  , '', 30,     -3,     3), '(#mu_{1}, #mu_{2}) #eta'                                       , 0)
histos['jpsi_phi'                  ] = (ROOT.RDF.TH1DModel('jpsi_phi'                  , '', 20, -np.pi, np.pi), '(#mu_{1}, #mu_{2}) #phi'                                       , 0)
histos['dr12'                      ] = (ROOT.RDF.TH1DModel('dr12'                      , '', 50,      0,   1.2), '#DeltaR(#mu_{1}, #mu_{2})'                                     , 0)
histos['dr13'                      ] = (ROOT.RDF.TH1DModel('dr13'                      , '', 50,      0,   1.2), '#DeltaR(#mu_{1}, #mu_{3})'                                     , 0)
histos['dr23'                      ] = (ROOT.RDF.TH1DModel('dr23'                      , '', 50,      0,   1.2), '#DeltaR(#mu_{2}, #mu_{3})'                                     , 0)
#histos['dr_jpsi_mu'                ] = (ROOT.RDF.TH1DModel('dr_jpsi_mu'                , '', 50,      0,   1.2), '#DeltaR(J/#Psi, #mu_{3})'                                      , 0)
histos['dr_jpsi_mu'                ] = (ROOT.RDF.TH1DModel('dr_jpsi_mu'                , '', 50,      0,   3), '#DeltaR(J/#Psi, #mu_{3})'                                      , 0)
histos['maxdr'                     ] = (ROOT.RDF.TH1DModel('maxdr'                     , '', 50,      0,   1.2), 'maximum #DeltaR(#mu_{i}, #mu_{j})'                             , 0)
histos['mindr'                     ] = (ROOT.RDF.TH1DModel('mindr'                     , '', 50,      0,   1.2), 'minimum #DeltaR(#mu_{i}, #mu_{j})'                             , 0)
# histos['bdt_mu'                    ] = (ROOT.RDF.TH1DModel('bdt_mu'                    , '',100,      0,   1. ), 'BDT score #mu'                                                 , 1)
# histos['bdt_tau'                   ] = (ROOT.RDF.TH1DModel('bdt_tau'                   , '',100,      0,   1. ), 'BDT score #tau'                                                , 1)
# histos['bdt_bkg'                   ] = (ROOT.RDF.TH1DModel('bdt_bkg'                   , '',100,      0,   1. ), 'BDT score bkg'                                                 , 1)
histos['abs_mu1mu2_dz'             ] = (ROOT.RDF.TH1DModel('abs_mu1mu2_dz'             , '',100,      0,   0.5), '|#mu_{1} d_{z} - #mu_{2} d_{z}| (cm)'                          , 1)
histos['abs_mu1k_dz'               ] = (ROOT.RDF.TH1DModel('abs_mu1k_dz'               , '',100,      0,   0.5), '|#mu_{1} d_{z} - #mu_{3} d_{z}| (cm)'                          , 1)
histos['abs_mu2k_dz'               ] = (ROOT.RDF.TH1DModel('abs_mu2k_dz'               , '',100,      0,   0.5), '|#mu_{2} d_{z} - #mu_{3} d_{z}| (cm)'                          , 1)
histos['jpsiK_pt'                  ] = (ROOT.RDF.TH1DModel('jpsiK_pt'                  , '', 50,      0,    50), '(J/#Psi + K^{+}) p_{T} (GeV)'                                  , 0)

histos['jpsiK_mass'                ] = (ROOT.RDF.TH1DModel('jpsiK_mass'                , '', 25,      3.5,     6.5), '(J/#Psi + K^{+}) mass (GeV)'                                   , 0)

#histos['jpsiK_mass'                ] = (ROOT.RDF.TH1DModel('jpsiK_mass'                , '',100,      5,     6), '(J/#Psi + K^{+}) mass (GeV)'                                   , 0)
histos['jpsiK_eta'                 ] = (ROOT.RDF.TH1DModel('jpsiK_eta'                 , '', 30,     -3,     3), '(J/#Psi + K^{+}) #eta'                                         , 0)
histos['jpsiK_phi'                 ] = (ROOT.RDF.TH1DModel('jpsiK_phi'                 , '', 20, -np.pi, np.pi), '(J/#Psi + K^{+}) #phi'                                         , 0)
histos['jpsipi_pt'                 ] = (ROOT.RDF.TH1DModel('jpsipi_pt'                 , '', 50,      0,    50), '(J/#Psi + #pi^{+}) p_{T} (GeV)'                                , 0)
histos['jpsipi_mass'               ] = (ROOT.RDF.TH1DModel('jpsipi_mass'               , '', 80,      3,     8), '(J/#Psi + #pi^{+}) mass (GeV)'                                 , 0)
histos['jpsipi_eta'                ] = (ROOT.RDF.TH1DModel('jpsipi_eta'                , '', 30,     -3,     3), '(J/#Psi + #pi^{+}) #eta'                                       , 0)
histos['jpsipi_phi'                ] = (ROOT.RDF.TH1DModel('jpsipi_phi'                , '', 20, -np.pi, np.pi), '(J/#Psi + #pi^{+}) #phi'                                       , 0)
histos['bct'                       ] = (ROOT.RDF.TH1DModel('bct'                       , '', 50,      0,  1e-1), 'ct (cm)'                                                       , 0)
# histos['npv_good'                  ] = (ROOT.RDF.TH1DModel('npv_good'                  , '', 70,      0,    70), '# good PV'                                                     , 0)
histos['nPV'                       ] = (ROOT.RDF.TH1DModel('nPV'                       , '', 70,      0,    70), '#PV'                                                           , 0)

histos['mmm_p4_par'                ] = (ROOT.RDF.TH1DModel('mmm_p4_par'                , '', 50,    0,   100), '3-#mu p_{#parallel} (GeV)'                                     , 0)

histos['mmm_p4_perp'               ] = (ROOT.RDF.TH1DModel('mmm_p4_perp'               , '', 50,      0,    15), '3-#mu p_{#perp}  (GeV)'                                        , 0)

histos['Bdir_eta'                  ] = (ROOT.RDF.TH1DModel('Bdir_eta'                  , '', 30,     -3,     3), 'B #eta from PV-SV direction'                                   , 0)
histos['Bdir_phi'                  ] = (ROOT.RDF.TH1DModel('Bdir_phi'                  , '', 20, -np.pi, np.pi), 'B #phi from PV-SV direction'                                   , 0)

histos['mcorr'                     ] = (ROOT.RDF.TH1DModel('mcorr'                     , '', 50,      4,    15), 'm_{corr} (GeV)'                                                , 0)

histos['decay_time_pv_jpsi'             ] = (ROOT.RDF.TH1DModel('decay_time_ps'             , '', 50,      0,     5e-12), 't (ps)'                                                        , 0)

histos['ip3d'                      ] = (ROOT.RDF.TH1DModel('ip3d'                      , '', 50,  -0.05,  0.05), '#mu_{3} IP3D(vtx_{J/#Psi}) (cm)'                               , 0)
histos['ip3d_corr'                      ] = (ROOT.RDF.TH1DModel('ip3d_corr'                      , '', 50,  -0.05,  0.05), 'corrected #mu_{3} IP3D(vtx_{J/#Psi}) (cm)'                               , 0)
histos['ip3d_e_corr_new'               ] = (ROOT.RDF.TH1DModel('ip3d_e_corr_new'               , '', 50,      0,  0.01), 'corrected #mu_{3} IP3D(vtx_{J/#Psi}) unc. (cm)'                , 0)

histos['ip3d_sig_dcorr'             ] = (ROOT.RDF.TH1DModel('ip3d_sig_dcorr'             , '', 50,     -5,     5), 'corrected #mu_{3} IP3D(vtx_{J/#Psi}) significance'             , 0)

histos['ip3d_sig_corr'             ] = (ROOT.RDF.TH1DModel('ip3d_sig_corr'             , '', 50,     -5,     5), 'corrected #mu_{3} IP3D(vtx_{J/#Psi}) significance'             , 0)
histos['ip3d_e'                    ] = (ROOT.RDF.TH1DModel('ip3d_e'                    , '', 50,      0,  0.01), '#mu_{3} IP3D(vtx_{J/#Psi}) unc. (cm)'                          , 0)

histos['ip3d_sig'                  ] = (ROOT.RDF.TH1DModel('ip3d_sig'                  , '', 20,     -5,     5), '#mu_{3} IP3D(vtx_{J/#Psi}) significance'                       , 0)

# histos['bdt_mu'                    ] = (ROOT.RDF.TH1DModel('bdt_mu'                    , '', 50,      0,     1), 'BDT #mu score'                                                , 0)
#histos['bdt_tau'                   ] = (ROOT.RDF.TH1DModel('bdt_tau'                   , '', 50,      0.03,     0.8), 'BDT #tau score'                                                , 0)
#histos['bdt_tau'                   ] = (ROOT.RDF.TH1DModel('bdt_tau'                   , '', 50,      0.,     1.), 'BDT #tau score'                                                , 0)
# histos['bdt_bkg'                   ] = (ROOT.RDF.TH1DModel('bdt_bkg'                   , '', 50,      0,     1), 'BDT bkg score'                                                 , 0)
# histos['decay_time'                ] = (ROOT.RDF.TH1DModel('decay_time'                , '', 50,      0,  1e-9), 't (s)'                                                         , 0)

#histos['bdt'                   ] = (ROOT.RDF.TH1DModel('bdt_v4'                   , '', 15,      0.,     0.95), 'BDT '                                                , 0)
#histos['bdt_tau_fakes_v6'                   ] = (ROOT.RDF.TH1DModel('bdt_tau_fakes_v6'                   , '', 15,      0.,     1), 'BDT #tau vs fakes'                                                , 0)
#histos['bdt_mu_v1'                   ] = (ROOT.RDF.TH1DModel('bdt_mu_v1'                   , '', 15,      0.,     1), 'BDT #mu vs fakes'                                               , 0)

'''
#histos['bdt_tau_mu_v2'                   ] = (ROOT.RDF.TH1DModel('bdt_tau_mu_v2'                   , '', 15,      0.,     1), 'BDT #tau vs #mu'                                                , 0)

histos['norm'                      ] = (ROOT.RDF.TH1DModel('norm'                      , '',  1,      0,     1), 'normalisation'                                                 , 0)
