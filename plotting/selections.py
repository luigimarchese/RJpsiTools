leading = '((mu1pt)*(mu1pt > mu2pt & mu1pt > kpt) + (mu2pt)*(mu2pt > mu1pt & mu2pt > kpt) + (kpt)*(kpt > mu2pt & kpt > mu1pt))'
trailing = '((mu1pt)*(mu1pt < mu2pt & mu1pt < kpt) + (mu2pt)*(mu2pt < mu1pt & mu2pt < kpt) + (kpt)*(kpt < mu2pt & kpt < mu1pt))'
subleading = '((mu1pt)*(mu1pt != '+leading+' & mu1pt != '+trailing+') + (mu2pt)*(mu2pt != '+leading+' & mu2pt != '+trailing+') + (kpt)*(kpt != '+leading+' & kpt != '+trailing+'))'

prepreselection = ' & '.join([
    leading +' > 6',
    subleading +' > 4',
    trailing + ' > 4',
    'mu2pt>4',
    'kpt>4',
    'abs(keta)<2.5',
    'abs(mu1eta)<2.5',
    'abs(mu2eta)<2.5',
    #'bvtx_svprob>1e-4',
    'jpsivtx_svprob>1e-2',
    'mu1_mediumID>0.5',
    'mu2_mediumID>0.5',
    'dr12>0.01',
    'dr13>0.01',
    'dr23>0.01',
    'abs(mu1_dz-mu2_dz)<0.2',
    #'!(abs(mu1_dz-k_dz)<0.2 & abs(mu2_dz-k_dz)<0.2 & abs(k_dxy)<0.05 & bvtx_svprob>1e-4)',
     'abs(mu1_dz-k_dz)<0.2',
     'abs(mu2_dz-k_dz)<0.2',
    'abs(k_dxy)<0.05',
    'abs(mu1_dxy)<0.05',
    'abs(mu2_dxy)<0.05',
    #'abs_Beta<0.5',
    #'k_mediumID>0.5',
    #'k_triggerLooseId>0.5',
    #'bvtx_chi2>4',
    'Bmass<10',
    'Bpt_reco<80',
    #'abs((fakerate_onlydata_62-fakerate_alpha_62*fakerate_onlymc_62)/(1-fakerate_alpha_62))<10',
    #'abs((fakerate_onlydata_63-fakerate_alpha_63*fakerate_onlymc_63)/(1-fakerate_alpha_63))<10',
    #'abs((fakerate_onlydata_64-fakerate_alpha_64*fakerate_onlymc_64)/(1-fakerate_alpha_64))<10',
    #'abs((fakerate_onlydata_68-fakerate_alpha_68*fakerate_onlymc_68)/(1-fakerate_alpha_68))<10',
    #'abs((fakerate_onlydata_66-fakerate_alpha_66*fakerate_onlymc_66)/(1-fakerate_alpha_66))<10',
    #'abs((fakerate_onlydata_67-fakerate_alpha_67*fakerate_onlymc_67)/(1-fakerate_alpha_67))<10',
    #'bvtx_svprob>0.6 & kpt>7 & jpsivtx_log10_lxy_sig>0.2 & Q_sq<5',
    #'Q_sq>5.5',
    #'ip3d_sig_dcorr>0',
    #'bvtx_cos2D<0.9985',
    #'kpt<7',
    #'k_raw_db_corr_iso03_rel<10',
    #'k_raw_db_corr_iso03_rel<0.2',
    #'ip3d_sig_dcorr>=-2 & ip3d_sig_dcorr<0',
    #'k_raw_db_corr_iso03_rel<0.2'
])

triggerselection = ' & '.join([
    'mu1_isFromMuT > 0.5',
    'mu2_isFromMuT>0.5',
    'mu1_isFromJpsi_MuT>0.5',
    'mu2_isFromJpsi_MuT>0.5',
    'k_isFromMuT>0.5',
])

etaselection = '(((((abs(mu1eta)<1) & (abs(mu2eta)>1)) | ((abs(mu1eta)>1) & (abs(mu2eta)<1))) & (abs(jpsivtx_fit_mass-3.0969)<0.07)) | ((abs(mu1eta)<1) & (abs(mu2eta)<1) & (abs(jpsivtx_fit_mass-3.0969)<0.05)) | ((abs(mu1eta)>1) & (abs(mu2eta)>1) & (abs(jpsivtx_fit_mass-3.0969)<0.1)))'

preselection_general = ' & '.join([prepreselection, triggerselection, etaselection])

preselection = ' & '.join([prepreselection, triggerselection, etaselection, 'Bmass<6.3 '])
#preselection = ' & '.join([prepreselection, triggerselection, etaselection, 'Bmass<6.3 & Q_sq>5.5'])
#preselection = ' & '.join([prepreselection, triggerselection, etaselection, 'Bmass>5.3 & Bmass<6.3 '])
preselection_mc = ' & '.join([preselection, 'abs(k_genpdgId)==13'])

preselection_hm = ' & '.join([prepreselection, triggerselection, etaselection, 'Bmass>6.3'])
preselection_hm_mc = ' & '.join([preselection_hm, 'abs(k_genpdgId)==13'])


common_sel = 'k_softMvaId<0.5'
pass_iso = 'k_raw_db_corr_iso03_rel<0.2'

pass_id = pass_iso +" & "+ common_sel
fail_id = '((!(%s)) & (%s))' % (pass_iso, common_sel)


# selections for the inverted case
common_sel_inv = 'k_raw_db_corr_iso03_rel<0.2'
pass_iso_inv = 'k_softMvaId>0.5'

pass_id_inv = pass_iso_inv +"&"+ common_sel_inv
fail_id_inv = '((!(%s)) & (%s))' % (pass_iso_inv, common_sel_inv)

