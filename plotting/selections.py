preselection = ' & '.join([
    'mu1pt>3'               ,
    'mu2pt>3'               ,
    'kpt>2.5'               ,
    'abs(mu1eta)<2.5'       ,
    'abs(mu2eta)<2.5'       ,
    'abs(keta)<2.5'         ,
#     'bvtx_svprob>1e-4'      ,
#     'jpsivtx_svprob>1e-2'   ,
#     'Bmass<6.3'             ,
    'mu1_mediumID>0.5'      ,
    'mu2_mediumID>0.5'      ,
    'Bpt_reco>15'           ,
    'dr12>0.01'             ,
    'dr13>0.01'             ,
    'dr23>0.01'             ,

    'abs(mu1_dz-mu2_dz)<0.2', # *
    'abs(mu1_dz-k_dz)<0.2'  , # *
    'abs(mu2_dz-k_dz)<0.2'  , # *
    'abs(k_dxy)<0.05'       , # *
    'abs(mu1_dxy)<0.05'     , # *
    'abs(mu2_dxy)<0.05'     , # *
#     'mu1pt>6'               ,
#     'mu2pt>6'               ,
#     'kpt>10'                 ,
#     'Bcos2D>0.995'          , # *
#     'bvtx_cos2D>0.995'      , # *
#     'abs(jpsiK_mass-5.27929)>0.060',
#     'abs(jpsipi_mass-5.27929)>0.060',
#     'abs(Beta)>1.5'         ,
#     'jpsi_pt<20',
    'm_miss_sq>0.5'         , # *
#     'b_iso03_rel<0.3'       , # *
    'abs(jpsi_vtx_fit_mass-3.0969)<0.1', # *

#     'Bmass>4',
#     'mmm_p4_par>10',
    
#     'bdt_bkg<0.3',
#     'bdt_bkg<0.5',
#     'Q_sq>7',
#     'm_miss_sq<7',
#     'abs(jpsi_eta)<1',
])

pass_id = 'k_mediumID>0.5'
fail_id = '(!(%s))' % pass_id

preselection_mc = ' & '.join([preselection, 'abs(k_genpdgId)==13'])
