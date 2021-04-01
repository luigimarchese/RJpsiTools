sample_names = [
    'jpsi_tau' ,
    'jpsi_mu'  ,
#     'jpsi_pi'  ,
    'chic0_mu' ,
    'chic1_mu' ,
    'chic2_mu' ,
    'jpsi_hc'  ,
    'hc_mu'    ,
    'psi2s_mu' ,
    'psi2s_tau',
#     'jpsi_3pi' ,
    #'fakes',
    'jpsi_x_mu'     ,
    'jpsi_x',
    'data'     ,
]

bc_weight = 0.52
fr = 0.19

#normalisation for the jpsi+X sample used for the fakes
hb_norm = 6.7 
#hb_norm = 5.5 

#normalisation for the jpsi+mu sample used for the comb bkg jpsi+mu
hbmu_norm = 8.5

weights = dict()
#weights['jpsi_tau' ] = 1
weights['jpsi_tau' ] = bc_weight 
weights['jpsi_mu'  ] = bc_weight 
weights['jpsi_pi'  ] = bc_weight
weights['psi2s_mu' ] = bc_weight
weights['chic0_mu' ] = bc_weight
weights['chic1_mu' ] = bc_weight
weights['chic2_mu' ] = bc_weight
weights['hc_mu'    ] = bc_weight
weights['psi2s_tau'] = bc_weight
weights['jpsi_3pi' ] = bc_weight
weights['jpsi_hc'  ] = bc_weight
#weights['fakes'    ] = fr/(1-fr) 
weights['jpsi_x'    ] =  hb_norm  
weights['jpsi_x_mu'     ] = hbmu_norm 
weights['data'     ] = 1.

titles = dict()
titles['jpsi_tau' ] = 'B_{c}#rightarrowJ/#Psi#tau'
titles['jpsi_mu'  ] = 'B_{c}#rightarrowJ/#Psi#mu'
titles['jpsi_x_mu'     ] = 'J/#Psi + #mu'
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
titles['fakes'     ] = 'fakes'
titles['jpsi_x'   ] = 'J/#Psi + X'
