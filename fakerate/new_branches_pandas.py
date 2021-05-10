import ROOT
from uproot_methods import TLorentzVectorArray

def to_define(df):
    df['ip3d_sig'                  ] = df['ip3d']/df['ip3d_e']
    df['decay_time_ps'             ] = df['decay_time'] * 1e12
    df['abs_mu1_dxy'               ] = abs(df['mu1_dxy'])
    df['abs_mu2_dxy'               ] = abs(df['mu2_dxy'])
    df['mu1_dxy_sig'               ] = abs(df['mu1_dxy']//df['mu1_dxyErr'])
    df['mu2_dxy_sig'               ] = abs(df['mu2_dxy']/df['mu2_dxyErr'])                                      
    df['abs_k_dxy'                 ] = abs(df['k_dxy'])                                                   
    df['k_dxy_sig'                 ] = abs(df['k_dxy']/df['k_dxyErr'])                                          
    df['abs_mu1_dz'                ] = abs(df['mu1_dz'])                                                  
    df['abs_mu2_dz'                ] = abs(df['mu2_dz'])                                                  
    df['mu1_dz_sig'                ] = abs(df['mu1_dz']/df['mu1_dzErr'])                                        
    df['mu2_dz_sig'                ] = abs(df['mu2_dz']/df['mu2_dzErr'])                                        
    df['abs_k_dz'                  ] = abs(df['k_dz'])                                                    
    df['k_dz_sig'                  ] = abs(df['k_dz']/df['k_dzErr'])                                            
    df['abs_mu1mu2_dz'             ] = abs(df['mu1_dz']-df['mu2_dz'])                                           
    df['abs_mu1k_dz'               ] = abs(df['mu1_dz']-df['k_dz'])                                             
    df['abs_mu2k_dz'               ] = abs(df['mu2_dz']-df['k_dz'])                                             
    #df['bvtx_log10_svprob'         ] = TMath::Log10(df['bvtx_svprob'])                                    
    #df['jpsivtx_log10_svprob'      ] = TMath::Log10(df['jpsivtx_svprob'])
    '''df['bvtx_log10_lxy'            ] = TMath::Log10(df['bvtx_lxy)'                                       
    df['jpsivtx_log10_lxy'         ] = TMath::Log10(df['jpsivtx_lxy)'                                    
    df['bvtx_log10_lxy_sig'        ] = TMath::Log10(df['bvtx_lxy_sig)'                                   
    df['jpsivtx_log10_lxy_sig'     ] = TMath::Log10(df['jpsivtx_lxy_sig)'                                
    '''

    df['b_iso03_rel'               ] = df['b_iso03']/df['Bpt']                                                  
    df['b_iso04_rel'               ] = df['b_iso04']/df['Bpt']                                                  
    df['k_iso03_rel'               ] = df['k_iso03']/df['kpt']                                                  
    df['k_iso04_rel'               ] = df['k_iso04']/df['kpt']                                                  
    df['mu1_iso03_rel'             ] = df['mu1_iso03']/df['mu1pt']                                              
    df['mu1_iso04_rel'             ] = df['mu1_iso04']/df['mu2pt']                                              
    df['mu2_iso03_rel'             ] = df['mu2_iso03']/df['mu2pt']                                              
    df['mu2_iso04_rel'             ] = df['mu2_iso04']/df['mu2pt']                                              
    mu1_p4 =  TLorentzVectorArray.from_ptetaphim(df['mu1pt'], df['mu1eta'], df['mu1phi'], df['mu1mass'])  
    mu2_p4 =  TLorentzVectorArray.from_ptetaphim(df['mu2pt'], df['mu2eta'], df['mu2phi'], df['mu2mass'])  
    mu3_p4 =  TLorentzVectorArray.from_ptetaphim(df['kpt'], df['keta'], df['kphi'], df['kmass'])  
    k_p4 =  TLorentzVectorArray.from_ptetaphim(df['kpt'], df['keta'], df['kphi'], 0.493677)  
    mmm_p4 = mu1_p4+mu2_p4+mu3_p4
    df['m12'                       ] = (mu1_p4+mu2_p4).mass
    df['m23'                       ] = (mu2_p4 + mu3_p4).mass
    df['m13'                       ] = (mu1_p4+mu3_p4).mass

    #df['bct'                       ] = df['bvtx_lxy*6.275/Bpt_reco'                                      
    #df['jpsiK_p4'                  ] = df['mu1_p4+mu2_p4+kaon_p4'                                        
    #df['jpsiK_mass'                ] = df['jpsiK_p4.mass()'                                              
    #df['jpsiK_pt'                  ] = df['jpsiK_p4.pt()'                                                
    #df['jpsiK_eta'                 ] = df['jpsiK_p4.eta()'                                               
    #df['jpsiK_phi'                 ] = df['jpsiK_p4.phi()'                                               
    #df['pion_p4'                   ] = df['ROOT::Math::PtEtaPhiMVector(kpt, keta, kphi, 0.13957018)'      # this is at the correct pion mass
    #df['jpsipi_p4'                 ] = df['mu1_p4+mu2_p4+pion_p4'                                        
    #df['jpsipi_mass'               ] = df['jpsipi_p4.mass()'                                             
    #df['jpsipi_pt'                 ] = df['jpsipi_p4.pt()'                                               
    #df['jpsipi_eta'                ] = df['jpsipi_p4.eta()'                                              
    #df['jpsipi_phi'                ] = df['jpsipi_p4.phi()'                                              
    jpsi_p4                   = mu1_p4+mu2_p4                                                
    df['jpsi_pt'                   ] = jpsi_p4.pt                                                 
    df['jpsi_eta'                  ] = jpsi_p4.eta                                                
    df['jpsi_phi'                  ] = jpsi_p4.phi                                                
    df['jpsi_mass'                 ] = jpsi_p4.mass                                               
    #df['q2jpsik'                   ] = df['(((jpsiK_p4 * 6.275/jpsiK_p4.mass()) - jpsi_p4).M2())'        
    #df['q2jpsipi'                  ] = df['(((jpsipi_p4 * 6.275/jpsipi_p4.mass()) - jpsi_p4).M2())'      
    #df['m2missjpsik'               ] = df['(((jpsiK_p4 * 6.275/jpsiK_p4.mass()) - jpsi_p4 - kaon_p4).M2())'  
    #df['m2missjpsipi'              ] = df['(((jpsipi_p4 * 6.275/jpsipi_p4.mass()) - jpsi_p4 - pion_p4).M2())'
    #df['dr12'                      ] = df['ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect( mu2_p4.Vect())' 
    #df['dr13'                      ] = df['ROOT::Math::VectorUtil::DeltaR(mu1_p4.Vect( mu3_p4.Vect())' 
    #df['dr23'                      ] = df['ROOT::Math::VectorUtil::DeltaR(mu2_p4.Vect( mu3_p4.Vect())' 
    #df['dr_jpsi_mu'                ] = df['ROOT::Math::VectorUtil::DeltaR(jpsi_p4.Vect( mu3_p4.Vect())'
    #df['norm'                      ] = df['0.5'                                                          
    # is there a better way?     
    #df['maxdr'                     ] = df['dr12*(dr12>dr13 & dr12>dr23) + dr13*(dr13>dr12 & dr13>dr23) + dr23*(dr23>dr12 & dr23>dr13)'
    #df['mindr'                     ] = df['dr12*(dr12<dr13 & dr12<dr23) + dr13*(dr13<dr12 & dr13<dr23) + dr23*(dr23<dr12 & dr23<dr13)'
    #df['pv_to_sv'                  ] = df['ROOT::Math::XYZVector((bvtx_vtx_x - pv_x (bvtx_vtx_y - pv_y (bvtx_vtx_z - pv_z))'
    #df['Bdirection'                ] = df['pv_to_sv/sqrt(pv_to_sv.Mag2())'                                   
    #df['Bdir_eta'                  ] = df['Bdirection.eta()'                                             
    #df['Bdir_phi'                  ] = df['Bdirection.phi()'                                             
    #df['mmm_p4_par'                ] = df['mmm_p4.Vect().Dot(Bdirection)'                                
    #df['mmm_p4_perp'               ] = df['sqrt(mmm_p4.Vect().Mag2()-mmm_p4_par*mmm_p4_par)'             
    #df['mcorr'                     ] = df['sqrt(mmm_p4.mass()*mmm_p4.mass() + mmm_p4_perp*mmm_p4_perp) + mmm_p4_perp' # Eq. 3 https://cds.cern.ch/record/2697350/files/1910.13404.pdf
    
    
    return df
