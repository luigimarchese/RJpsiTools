leading = '((mu1pt)*(mu1pt > mu2pt & mu1pt > kpt) + (mu2pt)*(mu2pt > mu1pt & mu2pt > kpt) + (kpt)*(kpt > mu2pt & kpt > mu1pt))'
trailing = '((mu1pt)*(mu1pt < mu2pt & mu1pt < kpt) + (mu2pt)*(mu2pt < mu1pt & mu2pt < kpt) + (kpt)*(kpt < mu2pt & kpt < mu1pt))'
subleading = '((mu1pt)*(mu1pt != '+leading+' & mu1pt != '+trailing+') + (mu2pt)*(mu2pt != '+leading+' & mu2pt != '+trailing+') + (kpt)*(kpt != '+leading+' & kpt != '+trailing+'))'

prepreselection = ' & '.join([
    leading +' > 6',
    subleading +' > 4',
    trailing + ' > 4',
    #'(mu1pt)*!(mu1pt > mu2pt & mu1pt > kpt)>4',
    #'mu1pt>6',
    'mu2pt>4',
    'kpt>4',
    #'kpt>4',
    'abs(mu1eta)<2.5',
    'abs(mu2eta)<2.5',
    'abs(keta)<2.5',
    'bvtx_svprob>1e-4',
    'jpsivtx_svprob>1e-2',
    #'Bmass>6.3',
    #'Bmass>5',
    #'Bmass<10.',
    'mu1_mediumID>0.5',
    'mu2_mediumID>0.5',
    #'mu1_mediumID',
    #'mu2_mediumID',
    #'Bpt_reco>15',
    'dr12>0.01',
    'dr13>0.01',
    'dr23>0.01',
    'abs(mu1_dz-mu2_dz)<0.2',
    'abs(mu1_dz-k_dz)<0.2',
    'abs(mu2_dz-k_dz)<0.2',
    'abs(k_dxy)<0.05',
    'abs(mu1_dxy)<0.05',
    'abs(mu2_dxy)<0.05',
    #'bvtx_cos2D>0.995',
    #'m_miss_sq>0.5',
    'abs(jpsi_mass-3.0969)<0.1',
    #'mmm_p4_par>10',
    #'k_raw_db_corr_iso04_rel<0.3',
    #'jpsivtx_cos2D>0.99',
    'mu1_isFromMuT > 0.5',
    'mu2_isFromMuT>0.5',
    'mu1_isFromJpsi_MuT>0.5',
    'mu2_isFromJpsi_MuT>0.5',
    'k_isFromMuT>0.5',    
])

preselection = ' & '.join([prepreselection, 'Bmass<6.3'])
preselection_mc = ' & '.join([preselection, 'abs(k_genpdgId)==13'])

preselection_hm = ' & '.join([prepreselection, 'Bmass>6.3'])
preselection_hm_mc = ' & '.join([preselection_hm, 'abs(k_genpdgId)==13'])


pass_id = 'k_mediumID>0.5 & k_raw_db_corr_iso03_rel<0.2'
fail_id = '(!(%s))' % pass_id



