to_define = [
    ('bvtx_lxy_unc_corr'         , '1.14*bvtx_lxy_unc*(run==1) + bvtx_lxy_unc*(run>1)'            ),
    ('bvtx_lxy_sig_corr'         , 'bvtx_lxy/bvtx_lxy_unc_corr'                                   ),
    ('bvtx_log10_lxy_sig_corr'   , 'TMath::Log10(bvtx_lxy_sig_corr)'                              ),
    ('jpsivtx_lxy_unc_corr'      , '1.10*jpsivtx_lxy_unc*(run==1) + jpsivtx_lxy_unc*(run>1)'      ),
    ('jpsivtx_lxy_sig_corr'      , 'jpsivtx_lxy/jpsivtx_lxy_unc_corr'                             ),
    ('jpsivtx_log10_lxy_sig_corr', 'TMath::Log10(jpsivtx_lxy_sig_corr)'                           ),
    ('ip3d_e_corr'               , '1.15*ip3d_e*(run==1) + ip3d_e*(run>1)'                        ),
    ('ip3d_sig'                  , 'ip3d/ip3d_e'                                                  ),
    ('ip3d_sig_corr'             , 'ip3d/ip3d_e_corr'                                             ),
    ('decay_time_ps'             , 'decay_time*1e12'                                              ),
    ('abs_mu1_dxy'               , 'abs(mu1_dxy)'                                                 ),
    ('abs_mu2_dxy'               , 'abs(mu2_dxy)'                                                 ),
    ('mu1_dxy_sig'               , 'abs(mu1_dxy)/mu1_dxyErr'                                      ),
    ('mu2_dxy_sig'               , 'abs(mu2_dxy)/mu2_dxyErr'                                      ),
    ('abs_k_dxy'                 , 'abs(k_dxy)'                                                   ),
    ('k_dxy_sig'                 , 'abs(k_dxy)/k_dxyErr'                                          ),
    ('abs_mu1_dz'                , 'abs(mu1_dz)'                                                  ),
    ('abs_mu2_dz'                , 'abs(mu2_dz)'                                                  ),
    ('mu1_dz_sig'                , 'abs(mu1_dz)/mu1_dzErr'                                        ),
    ('mu2_dz_sig'                , 'abs(mu2_dz)/mu2_dzErr'                                        ),
    ('abs_k_dz'                  , 'abs(k_dz)'                                                    ),
    ('k_dz_sig'                  , 'abs(k_dz)/k_dzErr'                                            ),
    ('abs_mu1mu2_dz'             , 'abs(mu1_dz-mu2_dz)'                                           ),
    ('abs_mu1k_dz'               , 'abs(mu1_dz-k_dz)'                                             ),
    ('abs_mu2k_dz'               , 'abs(mu2_dz-k_dz)'                                             ),
    ('bvtx_log10_svprob'         , 'TMath::Log10(bvtx_svprob)'                                    ),
    ('jpsivtx_log10_svprob'      , 'TMath::Log10(jpsivtx_svprob)'                                 ),
    ('bvtx_log10_lxy'            , 'TMath::Log10(bvtx_lxy)'                                       ),
    ('jpsivtx_log10_lxy'         , 'TMath::Log10(jpsivtx_lxy)'                                    ),
    ('bvtx_log10_lxy_sig'        , 'TMath::Log10(bvtx_lxy_sig)'                                   ),
    ('jpsivtx_log10_lxy_sig'     , 'TMath::Log10(jpsivtx_lxy_sig)'                                ),
    ('b_iso03_rel'               , 'b_iso03/Bpt'                                                  ),
    ('b_iso04_rel'               , 'b_iso04/Bpt'                                                  ),
    ('k_iso03_rel'               , 'k_iso03/kpt'                                                  ),
    ('k_iso04_rel'               , 'k_iso04/kpt'                                                  ),
    ('mu1_iso03_rel'             , 'mu1_iso03/mu1pt'                                              ),
    ('mu1_iso04_rel'             , 'mu1_iso04/mu2pt'                                              ),
    ('mu2_iso03_rel'             , 'mu2_iso03/mu2pt'                                              ),
    ('mu2_iso04_rel'             , 'mu2_iso04/mu2pt'                                              ),
    ('mu1_p4'                    , 'ROOT::Math::PtEtaPhiMVector(mu1pt, mu1eta, mu1phi, mu1mass)'  ),
    ('mu2_p4'                    , 'ROOT::Math::PtEtaPhiMVector(mu2pt, mu2eta, mu2phi, mu2mass)'  ),
    ('mu3_p4'                    , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, kmass)'          ),
    ('kaon_p4'                   , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, 0.493677)'       ), # this is at the correct kaon mass
    ('mmm_p4'                    , 'mu1_p4+mu2_p4+mu3_p4'                                         ),
    ('bct'                       , 'bvtx_lxy*6.275/Bpt_reco'                                      ),
    ('jpsiK_p4'                  , 'mu1_p4+mu2_p4+kaon_p4'                                        ),
    ('jpsiK_mass'                , 'jpsiK_p4.mass()'                                              ),
    ('jpsiK_pt'                  , 'jpsiK_p4.pt()'                                                ),
    ('jpsiK_eta'                 , 'jpsiK_p4.eta()'                                               ),
    ('jpsiK_phi'                 , 'jpsiK_p4.phi()'                                               ),
    ('pion_p4'                   , 'ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, 0.13957018)'     ), # this is at the correct pion mass
    ('jpsipi_p4'                 , 'mu1_p4+mu2_p4+pion_p4'                                        ),
    ('jpsipi_mass'               , 'jpsipi_p4.mass()'                                             ),
    ('jpsipi_pt'                 , 'jpsipi_p4.pt()'                                               ),
    ('jpsipi_eta'                , 'jpsipi_p4.eta()'                                              ),
    ('jpsipi_phi'                , 'jpsipi_p4.phi()'                                              ),
    ('jpsi_p4'                   , 'mu1_p4+mu2_p4'                                                ),
    ('jpsi_pt'                   , 'jpsi_p4.pt()'                                                 ),
    ('jpsi_eta'                  , 'jpsi_p4.eta()'                                                ),
    ('jpsi_phi'                  , 'jpsi_p4.phi()'                                                ),
    ('jpsi_mass'                 , 'jpsi_p4.mass()'                                               ),
    ('q2jpsik'                   , '(((jpsiK_p4 * 6.275/jpsiK_p4.mass()) - jpsi_p4).M2())'        ),
    ('q2jpsipi'                  , '(((jpsipi_p4 * 6.275/jpsipi_p4.mass()) - jpsi_p4).M2())'      ),
    ('m2missjpsik'               , '(((jpsiK_p4 * 6.275/jpsiK_p4.mass()) - jpsi_p4 - kaon_p4).M2())'  ),
    ('m2missjpsipi'              , '(((jpsipi_p4 * 6.275/jpsipi_p4.mass()) - jpsi_p4 - pion_p4).M2())'),
    ('dr12'                      , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu2_p4.Vect())' ),
    ('dr13'                      , 'ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect(), mu3_p4.Vect())' ),
    ('dr23'                      , 'ROOT::Math::VectorUtil::DeltaR(mu2_p4.Vect(), mu3_p4.Vect())' ),
    ('dr_jpsi_mu'                , 'ROOT::Math::VectorUtil::DeltaR(jpsi_p4.Vect(), mu3_p4.Vect())'),
    ('norm'                      , '0.5'                                                          ),
    # is there a better way?     
    ('maxdr'                     , 'dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'),
    ('mindr'                     , 'dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'),
    ('pv_to_sv'                  , 'ROOT::Math::XYZVector((bvtx_vtx_x - pv_x), (bvtx_vtx_y - pv_y), (bvtx_vtx_z - pv_z))'),
    ('Bdirection'                , 'pv_to_sv/sqrt(pv_to_sv.Mag2())'                               ),    
    ('Bdir_eta'                  , 'Bdirection.eta()'                                             ),
    ('Bdir_phi'                  , 'Bdirection.phi()'                                             ),
    ('mmm_p4_par'                , 'mmm_p4.Vect().Dot(Bdirection)'                                ),
    ('mmm_p4_perp'               , 'sqrt(mmm_p4.Vect().Mag2()-mmm_p4_par*mmm_p4_par)'             ),
    ('mcorr'                     , 'sqrt(mmm_p4.mass()*mmm_p4.mass() + mmm_p4_perp*mmm_p4_perp) + mmm_p4_perp'), # Eq. 3 https://cds.cern.ch/record/2697350/files/1910.13404.pdf
]
